# eval/run_batch.py
import argparse
from datetime import datetime
from pathlib import Path

from .utils_io import read_jsonl, write_jsonl
from rag_src.chat_backends import run_lab_rag_with_trace, run_general_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="questions jsonl path")
    ap.add_argument("--out", required=True, help="output run jsonl path")
    ap.add_argument("--pipeline", choices=["lab_rag", "chat"], default="lab_rag")
    ap.add_argument("--model_name", required=True)  # <- model_name 해결
    args = ap.parse_args()

    run_id = datetime.now().isoformat()

    rows = []
    for item in read_jsonl(args.questions):
        q = item["question"]
        ref = item.get("reference")
        

        pipeline = item.get("pipeline, args.pipeline")

        if pipeline == "lab_rag":
            answer, contexts = run_lab_rag_with_trace(q)
            retrieval_meta = {
                "k": 4,
                "vectorstore": "chroma",
                "embedding_model": "intfloat/multilingual-e5-base",
                "chunk_size": 700,
                "chunk_overlap": 100,
            }
        else:
            answer = run_general_chat(
                question=q            )
            contexts = []
            retrieval_meta = None

        rows.append({
            "run_id": run_id,
            "pipeline": pipeline,
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "retrieval": retrieval_meta,
            "reference" : ref,
            "model": {"provider": "ollama", "name": args.model_name},
        })

    write_jsonl(args.out, rows)
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
