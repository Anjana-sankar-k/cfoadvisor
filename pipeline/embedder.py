from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List

# ── Constants ────────────────────────────────────────────────
COLLECTION_NAME = "financial_data"  # unified name
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Load Model ──────────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)

# ── Qdrant Client ───────────────────────────────────────────
client = QdrantClient(":memory:")  # in-memory for Streamlit Cloud

# ── Embedding & Storage ─────────────────────────────────────
def embed_and_store(chunks: List[str]):
    """
    Embed text chunks and store them in Qdrant.
    Recreates the collection each time to start fresh.
    """
    embeddings = model.encode(chunks, show_progress_bar=False)

    # Recreate collection fresh each upload
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Prepare points for upsert
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": chunk})
        for chunk, emb in zip(chunks, embeddings)
    ]

    # Upsert points into Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Retrieval ───────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve top_k relevant chunks from Qdrant given a query.
    """
    # Embed the query
    query_vec = model.encode([query])[0].tolist()

    # Query Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k
    )

    # Extract text from retrieved points
    return [point.payload["text"] for point in results.points]