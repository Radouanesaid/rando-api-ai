from pathlib import Path

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GEMINI_API_KEY, GROQ_API_KEY

CHROMA_DIR = Path(__file__).parent.parent / "chroma_data"
COLLECTION_NAME = "rando_club"
TOP_K = 3

_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)


def retrieve(question: str, n_results: int = TOP_K) -> list[dict]:
    query_vector = _embeddings.embed_query(question)
    results = _collection.query(query_embeddings=[query_vector], n_results=n_results)
    return [
        {"id": doc_id, "text": doc, "type": meta["type"]}
        for doc_id, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]


SYSTEM_PROMPT = """Tu es l'assistant du site de Rando Club, un club de randonnée/vélo à Paris et en Île-de-France.

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous
- Si le contexte ne contient pas l'information demandée, réponds explicitement que tu ne sais pas, sans inventer
- Réponds en français, de façon concise et naturelle
- N'invente aucune date, lieu ou chiffre qui n'est pas dans le contexte

Contexte :
{context}
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{question}"),
    ]
)

_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.2)

_answer_chain = _prompt | _llm | StrOutputParser()


def ask(question: str, callbacks: list | None = None) -> dict:
    docs = retrieve(question)
    context = "\n".join(f"- [{d['type']}] {d['text']}" for d in docs)

    answer = _answer_chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": callbacks or []},
    )
    return {"answer": answer, "sources": [d["id"] for d in docs]}
