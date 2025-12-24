from __future__ import annotations
## only about chat tools

# chat_backends.py
"""
연구소 관련 대화(RAG)와 일반 대화(고성능 LLM)를 담당하는 모듈.
agent_llm.py의 @tool 함수들은 여기 함수들을 호출만 한다.
"""
import json
from pathlib import Path
from operator import itemgetter
from typing import Optional,List
from datetime import datetime, timezone

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chat_models import init_chat_model

# 🔹 일반 대화용
general_llm = ChatOllama(model="exaone3.5:7.8b")


# ===== 전역 캐시 =====
_RAG_VECTORSTORE = None
_RAG_RETRIEVER = None
_RAG_CHAIN = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent                      # .../your_project

RAG_DIR = PROJECT_ROOT / "rag_file"
RAG_DB_DIR = PROJECT_ROOT / "_rag_chroma_db"

# RAG 설정
EMBED_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4

####################
# 함수 tools
####################

def get_vectorstore() -> Chroma:
    global _RAG_VECTORSTORE
    if _RAG_VECTORSTORE is not None:
        return _RAG_VECTORSTORE

    # 1) 로드 시도
    vs = _load_vectorstore_if_exists()
    if vs is None:
        # 2) 없으면 빌드
        vs = build_rag_vectorstore()

    _RAG_VECTORSTORE = vs
    return _RAG_VECTORSTORE

def get_retriever():
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None:
        _RAG_RETRIEVER = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    return _RAG_RETRIEVER

def get_chain():
    global _RAG_CHAIN
    if _RAG_CHAIN is None:
        _RAG_CHAIN = _build_lab_rag_chain(get_retriever())
    return _RAG_CHAIN

def reset_rag_cache(reset_vectorstore: bool = False):
    """
    실험 중 파라미터(k/프롬프트/체인 등) 바꿀 때 캐시 초기화.
    vectorstore까지 초기화하면 다음 호출에서 로드/빌드를 다시 함.
    """
    global _RAG_VECTORSTORE, _RAG_RETRIEVER, _RAG_CHAIN
    _RAG_CHAIN = None
    _RAG_RETRIEVER = None
    if reset_vectorstore:
        _RAG_VECTORSTORE = None

def _retrieve_docs(retriever, query: str) -> List[Document]:
    """
    LangChain 버전 차이를 흡수하는 안전 retriever 호출.
    최신: retriever.invoke(query)
    구버전: retriever.get_relevant_documents(query)
    최후: retriever._get_relevant_documents(query) (private)
    """
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    if hasattr(retriever, "_get_relevant_documents"):
        return retriever._get_relevant_documents(query)
    raise AttributeError("Retriever has no supported retrieval method (invoke/get_relevant_documents/_get_relevant_documents).")


# retriever가 돌려준 document  리스트를 하나의 긴 문자열로 바꿔주는 함수!! (context format함수)
def _format_docs(docs: List[Document]) -> str:
    """retriever 결과 Document 리스트를 프롬프트용 문자열로 포맷."""
    if not docs:
        return "관련 연구소 문서를 찾지 못했습니다."

    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "")
        source_type = meta.get("source_type", "")

        header_bits = [f"[{i}]"]
        if source_type:
            header_bits.append(f"({source_type})")
        if src:
            header_bits.append(f"- {src}")

        header = " ".join(header_bits)
        parts.append(f"{header}\n{d.page_content}")

    return "\n\n".join(parts)


# ---------- 1) Indexing: ---------- # 
#-------------------------- loader ---------------------- #
def load_rag_corpus(rag_dir: str | Path) -> List[Document]:
    """
    rag_file 폴더 안의 corpus.json + .md 파일들을 모두 읽어서
    LangChain Document 리스트로 반환하는 loader.
    
    """
    rag_path = Path(rag_dir)

    if not rag_path.exists():
        raise FileNotFoundError(f"[RAG_LOADER] rag_dir not found: {rag_path}")

    all_docs: List[Document] = []

    # 1) Markdown 파일들 로드 (unstructured 말고 그냥 TextLoader 사용)
    md_loader = DirectoryLoader(
        str(rag_path),
        glob="**/*.md",
        show_progress=True,
        loader_cls=TextLoader,              # ★ 핵심: TextLoader로 강제
        loader_kwargs={"encoding": "utf-8"}
    )
    md_docs = md_loader.load()
    for d in md_docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source_type", "markdown")
        all_docs.append(d)

    # 2) corpus.json 로드
    json_path = rag_path / "corpus.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # corpus.json이 리스트 형태라고 가정
        if not isinstance(data, list):
            print("[RAG_LOADER] corpus.json is not a list. Adjust parsing logic if needed.")
            data = [data]

        json_docs: List[Document] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                # 형식이 예상과 다르면 스킵
                continue

            # text/content/body 중에서 실제 본문에 해당하는 키를 찾아본다.
            text = (
                item.get("text")
                or item.get("content")
                or item.get("body")
                or ""
            )
            if not text.strip():
                continue

            # 나머지 필드는 메타데이터로 사용
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("text", "content", "body")
            }
            metadata["source_type"] = "corpus_json"
            metadata["source"] = str(json_path)
            metadata.setdefault("index", idx)

            json_docs.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        print(f"[RAG_LOADER] corpus.json docs: {len(json_docs)}")
        all_docs.extend(json_docs)
    else:
        print(f"[RAG_LOADER] corpus.json not found in {rag_path}, only md files will be used.")

    print(f"[RAG_LOADER] total docs: {len(all_docs)}")
    return all_docs

#-------------------------- vectorstore  ---------------------- #

def build_rag_vectorstore() -> Chroma:
    # 1) JSON + MD 모두 로드
    docs = load_rag_corpus(RAG_DIR)

    # 2) 청크 나누기
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)

    # 3) 임베딩 + vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(RAG_DB_DIR),
    )
    vectorstore.persist()

    print("[LAB_RAG] vector store build 완료")
    return vectorstore


def _load_vectorstore_if_exists() -> Optional[Chroma]:
    """
    persist 디렉토리가 있으면 로드만 한다.
    (주의: 이 경로에 유효한 Chroma DB가 있어야 함)
    """
    if not RAG_DB_DIR.exists():
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(RAG_DB_DIR),
        embedding_function=embeddings,
    )
    print("[LAB_RAG] vector store LOAD 완료")
    return vectorstore




# ---------- 2) Retriever & RAG 체인 구성 ----------

def _build_lab_rag_chain(retriever):
    """
    question(str) -> answer(str) 형태로 동작하는 RAG 체인 생성.
    - retriever: rag_file 기반 (corpus.json + md)
    - prompt: '컨텍스트 안에서만 대답해라'
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "너는 KIST 및 연구실 안내 assistant이다. "
                    "사용자의 질문 문장을  `question`인자 그대로 넣어라"
                    "아래 컨텍스트 안의 정보만 사용해서 답변해라. "
                    "컨텍스트에 없으면 모른다고 말해라.\n\n"
                    "컨텍스트:\n{context}"
                ),
            ),
            ("human", "질문: {question}"),
        ]
    )

    llm = general_llm

    _RAG_CHAIN  = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnableLambda(
            lambda x: {
                "context": _format_docs(x["context"]),
                "question": x["question"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[LAB_RAG] rag_chain 생성 완료")
    return _RAG_CHAIN 

# ---------- 3) Entry points ----------

def run_lab_rag(question: str) -> str:
    """
    KIST/연구실 관련 질문에 대해 RAG 기반으로 답변을 생성하는 함수.

    - agent_llm.py 의 lab_chat 툴에서 이 함수를 호출한다.
    - 내부적으로 rag_chain (retriever + prompt + llm)을 사용.
    """

    print(f"[LAB_RAG] run_lab_rag 호출. question={question!r}")
    try:
        answer = get_chain().invoke({"question": question})
        print("[LAB_RAG] run_lab_rag.invoke 완료")
        return answer
    except Exception as e:
        print(f"[LAB_RAG] 오류: {e}")
        return "연구소 관련 RAG 답변을 생성하는 중 오류가 발생했습니다."

def run_lab_rag_with_trace(question: str) -> tuple[str, List[dict[str, any]]]:
    """
    평가/로그 목적: answer + retrieved docs(contexts)까지 반환
    """
    retriever = get_retriever()

    # 1) retrieve (버전 호환)
    docs = _retrieve_docs(retriever, question)

    # 2) generate
    answer = get_chain().invoke({"question": question})
    # 3) contexts serialize-friendly 변환
    contexts = []
    for d in docs:
        contexts.append({
            "text": d.page_content,
            "metadata": d.metadata or {}
        })

    return answer, contexts

def run_general_chat(question: str) -> str:
    """
    연구소와 무관한 일반 대화를 위한 LLM 호출 함수.
    """

    resp = general_llm.invoke(question)

    # ChatOllama 응답 포맷에 따라 content 꺼내기
    if hasattr(resp, "content"):
        return resp.content
    return str(resp)




# if __name__ == "__main__":
#     # 간단한 디버그용 질문들
#     test_questions = [
#         "kist는 언제 설립되었니?",
#         "이 연구실은 어떤 연구를 해?",
#         "장기현장실습 분위기 어땠어?",
#     ]

#     from pprint import pprint

#     for q in test_questions:
#         print("\n" + "=" * 80)
#         print(f"[TEST] 질문: {q}")
#         print("=" * 80)

#         answer = run_lab_rag(q)
#         print("[TEST] 답변:")
#         print(answer)