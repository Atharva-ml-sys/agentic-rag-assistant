"""
Guarded Agent - the LangGraph agent wrapped with production guardrails:

  1. Input validation   - reject empty/oversized/invalid input
  2. Max-step limit      - stop the agent from looping forever
                           (prevents 'excessive agency' = wasted cost/time)
  3. Output check        - make sure we return a sensible final answer

Guardrails make the agent safe and predictable, which is essential
before putting any agent into production.
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
# Knowledge base + tools (same as before)
# ----------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "guarded_kb"

knowledge = [
    "CloudDesk costs 499 rupees per user per month.",
    "CloudDesk offers a 30-day free trial with no credit card required.",
    "TechCorp was founded in 2015 in Bangalore, India.",
    "Customer support is available Monday to Friday, 9 AM to 6 PM IST.",
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


@tool
def search_docs(query: str) -> str:
    """Search TechCorp's knowledge base for company information."""
    qv = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=3
    ).points
    return "\n".join(f"- {r.payload['text']}" for r in results)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


tools = [search_docs, multiply]
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
agent = create_react_agent(llm, tools)


# ----------------------------------------------------------------------
# GUARDRAIL 1: Input validation
# ----------------------------------------------------------------------
MAX_INPUT_LENGTH = 1000

def validate_input(question):
    """Return an error message if input is invalid, else None."""
    if question is None or not question.strip():
        return "Error: empty question. Please ask something."
    if len(question) > MAX_INPUT_LENGTH:
        return f"Error: question too long (max {MAX_INPUT_LENGTH} characters)."
    return None  # input is valid


# ----------------------------------------------------------------------
# GUARDRAIL 2 + 3: run agent with a max-step limit, then check output
# ----------------------------------------------------------------------
MAX_STEPS = 6  # safety cap: agent can't loop more than this

def guarded_ask(question):
    print(f"USER: {question!r}")

    # --- Guardrail 1: validate input ---
    error = validate_input(question)
    if error:
        print(f"GUARDRAIL: {error}\n")
        return

    # --- Guardrail 2: run with recursion_limit (max steps) ---
    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            # recursion_limit caps how many steps the agent can take
            config={"recursion_limit": MAX_STEPS},
        )
        answer = result["messages"][-1].content
    except Exception as e:
        # if the agent hits the step limit or any error, fail gracefully
        print(f"GUARDRAIL: agent stopped safely ({type(e).__name__}). "
              f"Try rephrasing.\n")
        return

    # --- Guardrail 3: output check ---
    if not answer or not answer.strip():
        print("GUARDRAIL: agent produced an empty answer.\n")
        return

    print(f"AGENT: {answer}\n")


# ----------------------------------------------------------------------
# TEST the guardrails with good AND bad inputs
# ----------------------------------------------------------------------
print("=" * 65)

# normal valid question
guarded_ask("How much does CloudDesk cost?")

# empty input -> blocked by guardrail 1
guarded_ask("   ")

# oversized input -> blocked by guardrail 1
guarded_ask("spam " * 500)

# another valid question
guarded_ask("When was TechCorp founded?")