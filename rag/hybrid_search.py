"""
Hybrid Search - combines semantic (vector) search with keyword (BM25) search,
then merges the two result lists using Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Vector search understands meaning (synonyms, paraphrasing)
  - BM25 keyword search nails exact words, codes, names, IDs
  - Together they cover each other's weaknesses.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------
# 1. Knowledge base. Note the line with an exact error code (XR-4471) -
#    this is where keyword search shines and pure vector search struggles.
# ----------------------------------------------------------------------
document = """Qdrant is a vector database used for semantic search.
It stores embeddings and finds similar vectors very fast.

Python is a popular programming language for data science and AI.
It has many libraries like NumPy, Pandas, and PyTorch.

The Eiffel Tower is located in Paris, France.
It was built in 1889 and is a famous tourist attraction.

The error code XR-4471 means a disk failure has occurred.
Replace the faulty drive and restart the system to fix it."""

# ----------------------------------------------------------------------
# 2. Chunk the document
# ----------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
print(f"Created {len(chunks)} chunks\n")

# ----------------------------------------------------------------------
# 3. Build the VECTOR search side (Qdrant)
# ----------------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

client = QdrantClient(url="http://localhost:6333")
collection_name = "hybrid_demo"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
]
client.upsert(collection_name=collection_name, points=points)

# ----------------------------------------------------------------------
# 4. Build the KEYWORD search side (BM25)
#    BM25 works on tokenized text (split into lowercase words).
# ----------------------------------------------------------------------
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)


# ----------------------------------------------------------------------
# Helper: vector search -> returns a ranked list of chunk indices
# ----------------------------------------------------------------------
def vector_search(query, top_k=len(chunks)):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points
    # return chunk indices in ranked order (best first)
    return [r.id for r in results]


# ----------------------------------------------------------------------
# Helper: BM25 keyword search -> returns a ranked list of chunk indices
# ----------------------------------------------------------------------
def keyword_search(query, top_k=len(chunks)):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # sort chunk indices by score, highest first
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


# ----------------------------------------------------------------------
# 5. Reciprocal Rank Fusion (RRF) - merge the two ranked lists
#    Idea: ignore raw scores, use only rank position.
#    A chunk ranked high in BOTH lists bubbles to the top.
#    score(chunk) = sum over lists of 1 / (k + rank_in_that_list)
# ----------------------------------------------------------------------
def reciprocal_rank_fusion(list_a, list_b, k=60):
    scores = {}
    for ranked_list in (list_a, list_b):
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    # sort all chunk ids by their fused score, highest first
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ----------------------------------------------------------------------
# 6. Try a few queries and compare the three approaches
# ----------------------------------------------------------------------
def run_query(query):
    print("=" * 65)
    print(f"QUERY: {query}\n")

    vec_ranked = vector_search(query)
    kw_ranked = keyword_search(query)
    fused = reciprocal_rank_fusion(vec_ranked, kw_ranked)

    print(f"  Vector top pick : {chunks[vec_ranked[0]][:60]}...")
    print(f"  BM25 top pick   : {chunks[kw_ranked[0]][:60]}...")
    print(f"  HYBRID top pick : {chunks[fused[0][0]][:60]}...")
    print()


# A meaning-based query (vector should do well)
run_query("Where is the Eiffel Tower?")

# An exact-code query (BM25 should do well, vector may struggle)
run_query("XR-4471")