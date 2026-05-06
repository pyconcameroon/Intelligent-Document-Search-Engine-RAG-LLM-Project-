import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pypdf import PdfReader

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def get_embeddings():
    if EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
    return OpenAIEmbeddings()


def load_pdf_text(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Document(page_content=text, metadata={"source": f"{pdf_path.name} - page {page_number}"}))
    return pages


def create_vector_store(documents, persist_directory: Path):
    embeddings = get_embeddings()
    chroma = Chroma.from_documents(
        documents=documents,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    chroma.persist()
    return chroma


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF file into a local Chroma vector store.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--persist_dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="Directory where the local vector store will be persisted.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    persist_dir = Path(args.persist_dir).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf_text(pdf_path)
    if not docs:
        raise ValueError("No text was extracted from the PDF file.")

    print(f"Splitting {len(docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print(f"Creating vector store in: {persist_dir}")
    create_vector_store(chunks, persist_dir)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
