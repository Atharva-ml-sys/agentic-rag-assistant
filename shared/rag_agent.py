"""
Reusable RAG agent - the core agent logic in one place so the API
(and anything else) can import and use it without duplicating code.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ----------------------------------------------------------------------
# Knowledge base setup
# ----------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
            PointStruct(id=i, vector=embed_model.encode(t).tolist(),
                        payload={"text": t})
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
    qv = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=3
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

# Input guardrail config
MAX_INPUT_LENGTH = 1000
MAX_STEPS = 6


def ask_agent(question: str) -> str:
    """
    Main entry point: takes a question, applies guardrails, runs the
    agent, and returns the answer string. Used by the API.
    """
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
        return "Sorry, I couldn't process that. Please try rephrasing."

    # Guardrail 3: output check
    if not answer or not answer.strip():
        return "Sorry, I couldn't generate an answer."

    return answer