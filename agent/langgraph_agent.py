"""
LangGraph Agent - a proper agent built as a graph.

Graph structure:
    START -> agent (LLM thinks)
          -> if tool needed: tools node -> back to agent
          -> if no tool needed: END (final answer)

LangGraph handles the loop, the state (memory), and tool execution for us,
which also makes tool-calling far more reliable than our manual loop.
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
# Set up the knowledge base in Qdrant (so the search tool has data)
# ----------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "agent_kb"

knowledge = [
    "CloudDesk costs 499 rupees per user per month.",
    "TechCorp was founded in 2015 in Bangalore, India.",
    "We offer a 30-day free trial with no credit card required.",
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


# ----------------------------------------------------------------------
# 1. Define tools using the @tool decorator.
#    The docstring becomes the tool's description for the LLM - so write
#    it clearly. This is the LangChain/LangGraph way of declaring tools.
# ----------------------------------------------------------------------
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for any multiplication."""
    return a * b


@tool
def search_docs(query: str) -> str:
    """Search TechCorp's knowledge base for company information such as
    product pricing, free trial details, support hours, or company history."""
    qv = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=1
    ).points
    return results[0].payload["text"] if results else "No information found."


tools = [multiply, search_docs]

# ----------------------------------------------------------------------
# 2. Set up the LLM (via langchain-groq) and bind the tools to it
# ----------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# ----------------------------------------------------------------------
# 3. Build the agent graph in ONE line.
#    create_react_agent builds the whole START -> agent -> tools -> agent
#    -> END graph for us. (ReAct = Reason + Act, a standard agent pattern.)
# ----------------------------------------------------------------------
agent = create_react_agent(llm, tools)


# ----------------------------------------------------------------------
# 4. Helper to run the agent and print what happened at each step
# ----------------------------------------------------------------------
def run(question):
    print("=" * 65)
    print(f"USER: {question}\n")

    # The agent takes a list of messages and returns the full updated state
    result = agent.invoke({"messages": [("user", question)]})

    # Walk through every message to see the agent's reasoning + tool use
    for msg in result["messages"]:
        # tool calls requested by the LLM
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [tool call] {tc['name']}({tc['args']})")
        # results returned by a tool
        elif msg.type == "tool":
            print(f"  [tool result] {msg.content}")

    # The final message is the agent's answer
    print(f"\nFINAL ANSWER: {result['messages'][-1].content}\n")


# ----------------------------------------------------------------------
# 5. Test the same questions that broke our manual loop yesterday
# ----------------------------------------------------------------------
run("What is 499 multiplied by 12?")
run("How much does CloudDesk cost?")
run("Hi there, how are you today?")
run("What is the yearly cost of CloudDesk for one user?")