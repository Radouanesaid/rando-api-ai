import json
from pathlib import Path

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GEMINI_API_KEY

CORPUS_DIR = Path(__file__).parent / "corpus"
CHROMA_DIR = Path(__file__).parent / "chroma_data"
COLLECTION_NAME = "rando_club"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)


def load_corpus() -> list[dict]:
    entries = []
    for file in CORPUS_DIR.glob("*.json"):
        entries.extend(json.loads(file.read_text(encoding="utf-8")))
    return entries


def index():
    entries = load_corpus()
    print(f"{len(entries)} entrées à indexer")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    texts = [e["text"] for e in entries]
    vectors = embeddings.embed_documents(texts)

    collection.upsert(
        ids=[e["id"] for e in entries],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"type": e["type"]} for e in entries],
    )
    print(f"Collection '{COLLECTION_NAME}' à jour : {collection.count()} documents")


def test_query(question: str, n_results: int = 3):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    query_vector = embeddings.embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)

    print(f"\nQuestion : {question}")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['type']}] (distance={dist:.3f}) {doc}")


if __name__ == "__main__":
    index()
    test_query("une rando facile")
    test_query("qu'est-ce qui s'est passé à Fontainebleau ?")
    test_query("depuis quand existe le club ?")
