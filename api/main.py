"""
FastAPI backend - exposes the RAG agent as a web API.

Endpoints:
  GET  /          -> health check (is the server alive?)
  POST /ask       -> send a question, get the agent's answer
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shared.rag_agent import ask_agent

# Create the FastAPI application
app = FastAPI(title="Agentic RAG Assistant API")
# Allow the browser-based frontend to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production, restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Request body schema (Pydantic validates incoming data automatically)
# ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


# ----------------------------------------------------------------------
# Health check endpoint - confirms the server is running
# ----------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agentic RAG Assistant API is running"}


# ----------------------------------------------------------------------
# Main endpoint - takes a question, returns the agent's answer
# ----------------------------------------------------------------------
@app.post("/ask")
def ask(request: AskRequest):
    answer = ask_agent(request.question)
    return {"question": request.question, "answer": answer}