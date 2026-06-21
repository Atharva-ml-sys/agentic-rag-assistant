"""
Reusable RAG agent (deploy-ready version).

Uses FastEmbed for embeddings instead of sentence-transformers/PyTorch,
which makes the app much lighter and suitable for free-tier deployment.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ----------------------------------------------------------------------
# Embedding model via FastEmbed (lightweight, no PyTorch needed).
# BAAI/bge-small-en-v1.5 outputs 384-dim vectors, same as before,
# so our Qdrant collection config stays compatible.
# ----------------------------------------------------------------------
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed(text: str):
    """Return the embedding vector (list of floats) for a piece of text."""
    # FastEmbed returns a generator; we take the first (and only) result
    return list(embed_model.embed([text]))[0].tolist()


# ----------------------------------------------------------------------
# Qdrant connection (URL comes from env so it works locally AND in cloud)
# ----------------------------------------------------------------------
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION = "production_kb"

knowledge = [
    "CloudDesk is TechCorp's flagship project management tool.",
    "CloudDesk costs 499 rupees per user per month.",
    "CloudDesk offers a 30-day free trial with no credit card required.",
    "TechCorp was founded in 2015 in Bangalore, India.",
    "TechCorp builds cloud-based software for small businesses.",
    "Customer support is available Monday to Friday, 9 AM to 6 PM IST.",
    "You can reach support at support@techcorp.example.com.",
    "To reset your password, go to Settings and click 'Reset Password'.",
    "CloudDesk integrates with Slack, Google Calendar, and Trello.",
    "Enterprise plans include priority support and custom onboarding.",
]


def _setup_knowledge_base():
    """Create the collection and load documents (runs once at startup)."""
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=embed(t), payload={"text": t})
            for i, t in enumerate(knowledge)
        ],
    )


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
@tool
def search_docs(query: str) -> str:
    """Search TechCorp's knowledge base for company information such as
    product features, pricing, trial, integrations, support, or history."""
    results = qdrant.query_points(
        collection_name=COLLECTION, query=embed(query), limit=3
    ).points
    return "\n".join(f"- {r.payload['text']}" for r in results)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use for any multiplication."""
    return a * b


# ----------------------------------------------------------------------
# Build the agent (created once and reused)
# ----------------------------------------------------------------------
_setup_knowledge_base()
_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
_agent = create_react_agent(_llm, [search_docs, multiply])

# Guardrail config
MAX_INPUT_LENGTH = 1000
MAX_STEPS = 6


def ask_agent(question: str) -> str:
    """Apply guardrails, run the agent, return the answer. Used by the API."""
    # Guardrail 1: validate input
    if not question or not question.strip():
        return "Please ask a valid question."
    if len(question) > MAX_INPUT_LENGTH:
        return f"Question too long (max {MAX_INPUT_LENGTH} characters)."

    # Guardrail 2: run with a max-step limit
    try:
        result = _agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": MAX_STEPS},
        )
        answer = result["messages"][-1].content
    except Exception:
        return "I don't have enough information to answer that confidently."

    # Guardrail 3: output check
    if not answer or not answer.strip():
        return "Sorry, I couldn't generate an answer."

    return answer