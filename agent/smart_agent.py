"""
Smart Agent - LangGraph agent upgraded with:
  1. Real retrieval over a larger knowledge base
  2. Conversation memory (the agent remembers previous turns)

Memory works via a 'checkpointer' + a 'thread_id'. Each thread_id is like
a separate chat session whose full history the agent can recall.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ----------------------------------------------------------------------
# Build a richer knowledge base in Qdrant
# ----------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "smart_agent_kb"

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

if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
qdrant.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(id=i, vector=embed_model.encode(t).tolist(), payload={"text": t})
        for i, t in enumerate(knowledge)
    ],
)


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
@tool
def search_docs(query: str) -> str:
    """Search TechCorp's knowledge base for company info such as product
    features, pricing, trial, integrations, support, or company history.
    Returns the top 3 most relevant pieces of information."""
    qv = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=3
    ).points
    # join the top results into one context string
    return "\n".join(f"- {r.payload['text']}" for r in results)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for any multiplication."""
    return a * b


tools = [search_docs, multiply]

# ----------------------------------------------------------------------
# LLM + memory + agent
# ----------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# MemorySaver stores conversation state so the agent remembers past turns
memory = MemorySaver()

# Pass the checkpointer so the agent has memory
agent = create_react_agent(llm, tools, checkpointer=memory)


# ----------------------------------------------------------------------
# Helper: ask a question within a given conversation thread
# ----------------------------------------------------------------------
def ask(question, thread_id="default"):
    print(f"USER: {question}")

    # The config tells the agent WHICH conversation thread this belongs to.
    # Same thread_id = same memory = agent recalls earlier messages.
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [("user", question)]},
        config=config,
    )
    print(f"AGENT: {result['messages'][-1].content}\n")


# ----------------------------------------------------------------------
# DEMO: a multi-turn conversation to prove memory works
# ----------------------------------------------------------------------
print("=" * 65)
print("CONVERSATION (all in the same thread, so memory is shared)\n")

ask("How much does CloudDesk cost?")
# The next question says "it" and "yearly" - only answerable if the agent
# REMEMBERS the previous turn was about CloudDesk's monthly price.
ask("And what would that be yearly?")
# Another follow-up relying on memory
ask("Does it offer a free trial?")