# eval/ragas_eval.py
import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset as HFDataset
from .utils_io import read_jsonl,normalize_contexts


ENV_PATH = Path("/home/shy80/Desktop/KIST_AutonomousDriving/llm_control/STT_control/.env")
load_dotenv(dotenv_path=ENV_PATH, override=False)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        f"OPENAI_API_KEY not found. Check {ENV_PATH} and ensure it contains OPENAI_API_KEY=..."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["lab_rag", "chat"], required=True,
                    help="evaluation mode: lab_rag (RAG metrics) or chat (non-RAG metrics)")
    ap.add_argument("--run", required=True, help="run jsonl path")
    ap.add_argument("--out_csv", required=True, help="output csv path")
    ap.add_argument("--out_json", required=True, help="output json path")
    args = ap.parse_args()

    # 1) load run logs
    questions, answers, contexts_list,references = [], [], [], []
    for row in read_jsonl(args.run):
        questions.append(row["question"])
        answers.append(row["answer"])
        # RAGAS expects List[str] per sample
        contexts_list.append([c["text"] for c in row.get("contexts", [])])
        references.append(row.get("reference", "")) 

    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "reference": references,
    })

#### 이게 더 안전하다는데 일단 위의 버전으로 ㄱㄱ
###### contexts가 List[str]일수도 있고 List[dict]일수도 있어서.

# 1) load run logs
# rows = list(read_jsonl(args.run))  # ✅ rows 정의

# df = pd.DataFrame({
#     "question": [r["question"] for r in rows],
#     "answer":   [r["answer"] for r in rows],
#     "contexts": [normalize_contexts(r.get("contexts")) for r in rows],  # ✅
#     "reference":[r.get("reference", "") for r in rows],
# })
    # 2) ragas evaluate
    hf_ds = HFDataset.from_pandas(df)
    judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
)
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    )

    # 3) pipeline별 metrics 선택

    # result = evaluate(
    #     hf_ds,
    #     metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    #     llm=judge_llm,
    #     embeddings=judge_embeddings,
    # )
    if args.pipeline == "chat":
        # Non-RAG: context 기반 지표는 의미가 없으므로 제외
        metrics = [answer_relevancy]
        result = evaluate(hf_ds, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

        # 결과 변환
        try:
            result_df = result.to_pandas()
        except Exception:
            result_df = pd.DataFrame(result)

    else:
        # RAG: reference 유무에 따라 context_recall 포함 여부를 분리해 평가
        has_ref = df["reference"].fillna("").astype(str).str.strip().str.len() > 0

        parts = []

        # (A) reference 있는 샘플: context_recall 포함
        if has_ref.any():
            df_ref = df[has_ref].copy()
            hf_ref = HFDataset.from_pandas(df_ref)
            res_ref = evaluate(
                hf_ref,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
            try:
                parts.append(res_ref.to_pandas())
            except Exception:
                parts.append(pd.DataFrame(res_ref))

        # (B) reference 없는 샘플: context_recall 제외
        if (~has_ref).any():
            df_noref = df[~has_ref].copy()
            hf_noref = HFDataset.from_pandas(df_noref)
            res_noref = evaluate(
                hf_noref,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
            try:
                parts.append(res_noref.to_pandas())
            except Exception:
                parts.append(pd.DataFrame(res_noref))

        # 두 결과를 행 순서대로 합치기 위해 index를 보존하는 컬럼을 하나 둬야 하지만,
        # 여기서는 평가용 CSV/JSON 저장이 목적이므로 concat 후 question/answer로 구분 가능.
        # (원하면 idx 컬럼 추가 방식으로 더 안정적으로 만들 수 있음)
        result_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


    # 3) save reports
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # result는 ragas 버전에 따라 dict-like / dataframe-like 일 수 있어 변환 처리
    try:
        result_df = result.to_pandas()
    except Exception:
        # fallback
        result_df = pd.DataFrame(result)

    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    result_df.to_json(out_json, orient="records", force_ascii=False, indent=2)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_json}")

if __name__ == "__main__":
    main()
