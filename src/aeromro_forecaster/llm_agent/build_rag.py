from __future__ import annotations

import argparse
from pathlib import Path


def build_rag_index(docs_dir: Path, persist_directory: Path) -> int:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install langchain-community, langchain-ollama, chromadb, and pypdf to build RAG.") from exc

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    docs = []
    for pdf in pdfs:
        docs.extend(PyPDFLoader(str(pdf)).load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75).split_documents(docs)
    Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="mro_manuals",
        persist_directory=str(persist_directory),
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from local MRO PDFs.")
    parser.add_argument("--docs-dir", type=Path, default=Path("data/docs"))
    parser.add_argument("--persist-directory", type=Path, default=Path("data/chroma_db"))
    args = parser.parse_args()
    count = build_rag_index(args.docs_dir, args.persist_directory)
    print(f"Indexed {count:,} document chunks.")


if __name__ == "__main__":
    main()
