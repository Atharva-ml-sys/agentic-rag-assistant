"""
Multi-Tool Agent Demo - the LLM has TWO tools and decides which to use.

Tools:
  1. multiply      -> for math
  2. search_docs   -> searches our Qdrant knowledge base (real RAG retrieval)

Includes basic error handling so one bad tool-call doesn't crash everything -
this is important for building reliable agents in production.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ----------------------------------------------------------------------
# Set up a small knowledge base in Qdrant (so search_docs has data)
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
# 1. Define the two real tools
# ----------------------------------------------------------------------
def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def search_docs(query):
    """Search the company knowledge base and return the best matching text."""
    qv = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=1
    ).points
    return results[0].payload["text"] if results else "No information found."


available_tools = {"multiply": multiply, "search_docs": search_docs}

# ----------------------------------------------------------------------
# 2. Describe both tools to the LLM (clear descriptions help the model
#    produce a valid tool call)
# ----------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together. Use this for any "
                           "arithmetic multiplication.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "the first number"},
                    "b": {"type": "number", "description": "the second number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search TechCorp's knowledge base to find company "
                           "information such as product pricing, free trial "
                           "details, support hours, or company history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the search query, e.g. 'CloudDesk price'",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ----------------------------------------------------------------------
# 3. The agentic loop, with error handling around the LLM call
# ----------------------------------------------------------------------
def run_agent(user_question):
    print("=" * 65)
    print(f"USER: {user_question}\n")

    messages = [{"role": "user", "content": user_question}]

    for step in range(5):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except BadRequestError as e:
            # The model produced a malformed tool call. Instead of crashing,
            # we report it and stop gracefully.
            print(f"  [!] Tool call failed (model formatting issue): {e}\n")
            return

        msg = response.choices[0].message

        # No tool requested -> this is the final answer
        if not msg.tool_calls:
            print(f"FINAL ANSWER: {msg.content}\n")
            return

        # Run each requested tool
        messages.append(msg)
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = available_tools[name](**args)

            print(f"  [Step {step+1}] LLM chose tool: {name}({args})")
            print(f"            -> result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": str(result),
            })
        print()


# ----------------------------------------------------------------------
# 4. Test with different kinds of questions
# ----------------------------------------------------------------------
run_agent("What is 499 multiplied by 12?")          # needs multiply
run_agent("How much does CloudDesk cost?")           # needs search_docs
run_agent("Hi there, how are you today?")            # needs NO tool
run_agent("What is the yearly cost of CloudDesk for one user?")  # needs BOTH tools