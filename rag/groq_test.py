"""
Quick test to confirm the Groq API key works and we can call the LLM.
"""

import os
from dotenv import load_dotenv
from groq import Groq

# Load the GROQ_API_KEY from the .env file
load_dotenv()

# Create the Groq client (it automatically reads GROQ_API_KEY from env)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Send a simple test message to the LLM
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
)

print("LLM says:", response.choices[0].message.content)