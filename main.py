import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.vectorstores import Chroma
from pydantic import BaseModel
from transformers import pipeline

load_dotenv()

PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
HUGGINGFACE_LLM_MODEL = os.getenv("HUGGINGFACE_LLM_MODEL", "google/flan-t5-small")

app = FastAPI(title="Intelligent Doc Search", version="0.1.0")
qa_chain = None


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]


def get_embeddings():
    if EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
    return OpenAIEmbeddings()


def get_llm():
    if LLM_PROVIDER == "huggingface":
        text_gen = pipeline(
            "text2text-generation",
            model=HUGGINGFACE_LLM_MODEL,
            max_length=256,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=text_gen)

    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(temperature=0)

    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY or LLM_PROVIDER=huggingface."
    )


def load_vector_store():
    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        raise FileNotFoundError(
            f"Vector store not found in {PERSIST_DIR}. Run ingest.py first."
        )

    embeddings = get_embeddings()
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


@app.on_event("startup")
async def startup_event():
    global qa_chain
    try:
        vector_store = load_vector_store()
    except FileNotFoundError as exc:
        qa_chain = None
        app.state.startup_error = str(exc)
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    app.state.startup_error = None


@app.get("/")
async def root():
    return {
        "service": "Intelligent Doc Search",
        "status": "ready" if qa_chain else "waiting for ingest",
    }


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if app.state.startup_error:
        raise HTTPException(status_code=500, detail=app.state.startup_error)
    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail="The vector store is not loaded. Run ingest.py to index your PDF first.",
        )

    qa_chain.retriever.search_kwargs["k"] = request.k or 4
    result = qa_chain(request.question)
    answer = result.get("result")
    source_docs = result.get("source_documents", [])
    sources = []
    for doc in source_docs:
        source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        if source:
            sources.append(source)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=list(dict.fromkeys(sources)),
    )
