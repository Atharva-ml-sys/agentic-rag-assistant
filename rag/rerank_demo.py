"""
Re-ranking Demo - shows how a cross-encoder re-ranker improves the
ORDER of retrieved chunks.

Pipeline idea:
  fast retrieval (gets candidates, rough order)
      -> re-ranker (slow but smart, fixes the order)
      -> top results in the BEST possible order
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

# ----------------------------------------------------------------------
# Candidate chunks (imagine these came from hybrid search already)
# ----------------------------------------------------------------------
chunks = [
    "Python is a programming language used for AI and data science.",
    "The capital of France is Paris, a major European city.",
    "To reset your password, click 'Forgot Password' on the login page.",
    "Paris hosts the Eiffel Tower and many famous art museums.",
    "Our refund policy allows returns within 30 days of purchase.",
]

query = "How do I recover my account login?"

print("=" * 65)
print(f"QUERY: {query}\n")

# ----------------------------------------------------------------------
# STAGE 1: Fast retrieval using bi-encoder (embedding similarity)
# This is what we already built - fast, but order is approximate.
# ----------------------------------------------------------------------
print(">>> STAGE 1: Fast embedding search (bi-encoder)\n")

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = bi_encoder.encode(chunks)
query_embedding = bi_encoder.encode(query)

# score each chunk by cosine similarity
bi_scores = [cos_sim(query_embedding, ce).item() for ce in chunk_embeddings]

# sort chunks by score (highest first)
bi_ranked = sorted(zip(chunks, bi_scores), key=lambda x: x[1], reverse=True)

for chunk, score in bi_ranked:
    print(f"  {score:.3f}  |  {chunk}")

# ----------------------------------------------------------------------
# STAGE 2: Re-rank with a cross-encoder (slow but accurate)
# The cross-encoder reads query + chunk TOGETHER for each pair.
# ----------------------------------------------------------------------
print("\n>>> STAGE 2: Re-ranked with cross-encoder\n")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# build (query, chunk) pairs for every candidate
pairs = [[query, chunk] for chunk in chunks]
cross_scores = cross_encoder.predict(pairs)

# sort chunks by cross-encoder score (highest first)
cross_ranked = sorted(zip(chunks, cross_scores), key=lambda x: x[1], reverse=True)

for chunk, score in cross_ranked:
    print(f"  {score:6.2f}  |  {chunk}")

# ----------------------------------------------------------------------
# Compare the #1 pick from each stage
# ----------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"Bi-encoder top pick   : {bi_ranked[0][0]}")
print(f"Cross-encoder top pick: {cross_ranked[0][0]}")