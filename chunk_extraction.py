
import tiktoken
from pathlib import Path
from typing import List, Dict

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_markdown_by_file(md_path: str) -> List[Dict]:
    chunks = []
    current_doc = None
    current_lines = []

    with open(md_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip().startswith("# file: ") and not line.strip().startswith("## "):
                if current_doc:
                    chunks.append({
                        "doc_id": current_doc,
                        "text": "".join(current_lines).strip()
                    })
                    current_lines = []
                current_doc = line.strip("# ").strip()
            else:
                current_lines.append(line)

        if current_doc and current_lines:
            chunks.append({
                "doc_id": current_doc,
                "text": "".join(current_lines).strip()
            })

    return chunks

from config import Configurator
if __name__ == "__main__":
    cfg = Configurator("config.yaml")
    md_file = cfg.output_md_path
    chunks = chunk_markdown_by_file(md_file)

    for chunk in chunks:
        chunk["num_tokens"] = count_tokens(chunk["text"])

    top_chunks = sorted(chunks, key=lambda x: x["num_tokens"], reverse=True)[:5]

    for c in top_chunks:
        print(f"{c['doc_id']}: {c['num_tokens']} tokens")
