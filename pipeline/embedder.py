from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

COLLECTION_NAME = "financials"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(":memory:")  # in-memory, resets each session — fine for V0


def embed_and_store(chunks: list[str]):
    embeddings = model.encode(chunks, show_progress_bar=False)

    # Recreate collection fresh each upload
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": chunk})
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query: str, top_k: int = 5) -> list[str]:
    query_vec = model.encode([query])[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )
    return [r.payload["text"] for r in results]