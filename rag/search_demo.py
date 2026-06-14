"""
Search Demo - The complete 'Retrieval' part of RAG:
document -> chunks -> embeddings -> store in Qdrant -> search!
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ----------------------------------------------------------------------
# 1. Our document (acts as a small knowledge base)
# ----------------------------------------------------------------------
document = """Qdrant is a vector database used for semantic search.
It stores embeddings and finds similar vectors very fast.

Python is a popular programming language for data science and AI.
It has many libraries like NumPy, Pandas, and PyTorch.

The Eiffel Tower is located in Paris, France.
It was built in 1889 and is a famous tourist attraction.

Machine learning models learn patterns from data.
They are trained on large datasets to make predictions."""

# ----------------------------------------------------------------------
# 2. Chunking - split the document into smaller pieces (recursive = smart)
# ----------------------------------------------------------------------
print("Step 1: Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
print(f"  -> {len(chunks)} chunks created\n")

# ----------------------------------------------------------------------
# 3. Embedding - convert each chunk into a 384-number vector
# ----------------------------------------------------------------------
print("Step 2: Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print(f"  -> {len(embeddings)} embeddings created (each {len(embeddings[0])}-dim)\n")

# ----------------------------------------------------------------------
# 4. Connect to Qdrant and create a 'collection'
#    (a collection is like a table where vectors are stored)
# ----------------------------------------------------------------------
print("Step 3: Storing in Qdrant...")
client = QdrantClient(url="http://localhost:6333")

collection_name = "my_first_collection"

# Create a fresh collection (delete first if it already exists)
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    # size=384 because our embedding model outputs 384 numbers
    # COSINE = the method used to measure how 'close' two vectors are
)

# Wrap each chunk into a 'point' and insert it into Qdrant
points = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    points.append(
        PointStruct(
            id=i,                          # unique id for each point
            vector=embedding.tolist(),     # the 384 numbers
            payload={"text": chunk},       # store the original text alongside
        )
    )

client.upsert(collection_name=collection_name, points=points)
print(f"  -> {len(points)} points stored in Qdrant\n")

# ----------------------------------------------------------------------
# 5. SEARCH! - ask a question, retrieve relevant chunks
# ----------------------------------------------------------------------
print("=" * 60)
query = "Where is the Eiffel Tower?"
print(f"QUERY: {query}\n")

# Convert the question into an embedding too (so we can compare)
query_vector = model.encode(query).tolist()

# Ask Qdrant for the top 2 most similar chunks
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2,
).points

print("TOP 2 RELEVANT CHUNKS (best match first):\n")
for r in results:
    print(f"  Score: {r.score:.3f}")
    print(f"  Text:  {r.payload['text']}")
    print()