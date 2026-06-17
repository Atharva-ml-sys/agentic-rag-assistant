"""
Full RAG Pipeline - the complete flow:

  document -> chunk -> embed -> store in Qdrant
  query    -> retrieve relevant chunks -> feed to LLM -> grounded answer

This is where Retrieval ("R") meets Generation ("G").
"""

import os
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load GROQ_API_KEY from .env
load_dotenv()

# ----------------------------------------------------------------------
# 1. Knowledge base (imagine this is a company's documentation)
# ----------------------------------------------------------------------
document = """Our company TechCorp was founded in 2015 in Bangalore, India.
We build cloud-based software for small businesses.

Our flagship product is CloudDesk, a project management tool.
CloudDesk costs 499 rupees per user per month.

To reset your password, go to Settings and click 'Reset Password'.
A reset link will be sent to your registered email within 5 minutes.

Our customer support is available Monday to Friday, 9 AM to 6 PM IST.
You can reach support at support@techcorp.example.com.

We offer a 30-day free trial for all new CloudDesk accounts.
No credit card is required to start the trial."""

# ----------------------------------------------------------------------
# 2. INDEXING: chunk -> embed -> store in Qdrant
#    (This is the "ingestion" part, done once when documents arrive.)
# ----------------------------------------------------------------------
print("Indexing the document...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)

client = QdrantClient(url="http://localhost:6333")
collection_name = "techcorp_docs"

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
print(f"  -> {len(chunks)} chunks indexed\n")


# ----------------------------------------------------------------------
# 3. RETRIEVAL: given a query, fetch the most relevant chunks
# ----------------------------------------------------------------------
def retrieve(query, top_k=3):
    query_vector = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points
    return [r.payload["text"] for r in results]


# ----------------------------------------------------------------------
# 4. GENERATION: feed query + retrieved context to the LLM
#    The prompt forces the LLM to answer ONLY from the given context.
# ----------------------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query, context_chunks):
    # Join the retrieved chunks into one context block
    context = "\n\n".join(context_chunks)

    # This prompt is the heart of RAG - it grounds the LLM in our data
    prompt = f"""You are a helpful assistant for TechCorp.
Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature = factual, less creative
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# 5. THE FULL RAG FLOW: ask questions and get grounded answers
# ----------------------------------------------------------------------
def ask(query):
    print("=" * 65)
    print(f"QUESTION: {query}\n")

    # Step A: retrieve relevant chunks
    chunks = retrieve(query)
    print("Retrieved context:")
    for c in chunks:
        print(f"  - {c}")

    # Step B: generate an answer grounded in those chunks
    answer = generate_answer(query, chunks)
    print(f"\nANSWER: {answer}\n")


# Try a few questions
ask("How much does CloudDesk cost?")
ask("How do I reset my password?")
ask("Who is the CEO of TechCorp?")   # not in the docs - should say "I don't know"