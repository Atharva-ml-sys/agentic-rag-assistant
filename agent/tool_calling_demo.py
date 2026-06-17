"""
Tool Calling Demo - shows how an LLM can decide to call a function (tool)
on its own, instead of guessing the answer.

Flow:
  user question -> LLM decides a tool is needed -> we run the tool
  -> give result back to LLM -> LLM gives final answer
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ----------------------------------------------------------------------
# 1. Define the actual tool (a normal Python function)
# ----------------------------------------------------------------------
def multiply(a, b):
    """Multiply two numbers."""
    return a * b


# ----------------------------------------------------------------------
# 2. Describe the tool to the LLM in a format it understands.
#    This tells the LLM: "you have a tool called multiply, here's what
#    it does and what inputs it needs."
# ----------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "first number"},
                    "b": {"type": "number", "description": "second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

# A registry so we can look up the real function by name
available_tools = {"multiply": multiply}


# ----------------------------------------------------------------------
# 3. Ask a question that needs the tool
# ----------------------------------------------------------------------
user_question = "What is 499 multiplied by 12?"
print(f"USER: {user_question}\n")

messages = [{"role": "user", "content": user_question}]

# First call: send the question + tell the LLM which tools it has
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,             # <-- here we hand the LLM its tools
    tool_choice="auto",      # <-- let the LLM decide if it needs a tool
)

response_message = response.choices[0].message

# ----------------------------------------------------------------------
# 4. Did the LLM decide to call a tool?
# ----------------------------------------------------------------------
if response_message.tool_calls:
    print(">>> LLM decided to use a tool!\n")

    # add the LLM's tool-call message to the conversation history
    messages.append(response_message)

    # run each tool the LLM asked for
    for tool_call in response_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"    Tool requested : {tool_name}")
        print(f"    Arguments      : {tool_args}")

        # actually run the real Python function
        result = available_tools[tool_name](**tool_args)
        print(f"    Tool result    : {result}\n")

        # give the result back to the LLM
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(result),
            }
        )

    # ------------------------------------------------------------------
    # 5. Second call: LLM now has the tool result, so it writes the
    #    final natural-language answer.
    # ------------------------------------------------------------------
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    print(f"FINAL ANSWER: {final_response.choices[0].message.content}")

else:
    # LLM answered directly without any tool
    print(f"FINAL ANSWER (no tool): {response_message.content}")