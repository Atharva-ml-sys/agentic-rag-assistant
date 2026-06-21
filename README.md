# Agentic RAG Assistant

A production-grade agentic Retrieval-Augmented Generation (RAG) system that answers questions from a private knowledge base, decides on its own which tools to use, executes multi-step reasoning, and stays grounded in retrieved documents instead of hallucinating. The full stack — vector database, agent, and API — is containerized and deployed live.

**Live API:** https://agentic-rag-assistant-production.up.railway.app
**Interactive docs:** https://agentic-rag-assistant-production.up.railway.app/docs

## Why this project

A plain LLM has two problems in production: it has no access to private or company-specific data, and it confidently makes things up when it doesn't know an answer. This system addresses both. Every answer is grounded in documents retrieved from a vector store, the agent reports when information isn't available rather than inventing it, and the whole pipeline is built with the engineering practices a real deployment needs — configurable services, guardrails, evaluation, and containerized deployment.

## How it works

The request flows through four stages:

```
            Documents
                |
                v
   chunk  ->  embed  ->  store in Qdrant (vector DB)
                                |
   User question --------------> FastAPI /ask
                                |
                                v
                   +-------------------------------+
                   |  LangGraph agent              |
                   |   - decides which tool to use |
                   |   - searches the knowledge base|
                   |   - runs multi-step reasoning |
                   |   - applies guardrails        |
                   +-------------------------------+
                                |
                                v
                   Groq LLM  ->  grounded answer
```

1. **Ingestion.** Documents are split into overlapping chunks and converted into vector embeddings (via FastEmbed), then stored in Qdrant.
2. **Retrieval.** A query is embedded and matched against the stored vectors to fetch the most relevant context. The retrieval experiments in this repo also cover hybrid search (dense vectors combined with BM25 keyword search, fused using Reciprocal Rank Fusion) and cross-encoder re-ranking.
3. **Agent.** A LangGraph agent reasons over the query, autonomously decides whether to call the document-search tool or the calculator tool, chains multiple steps when needed, and keeps conversation memory across turns.
4. **Generation.** The Groq-hosted LLM produces the final answer using the retrieved context.

## Key features

- Retrieval pipeline with hybrid search and cross-encoder re-ranking
- LangGraph agent with autonomous tool selection and multi-step reasoning
- Conversation memory for multi-turn questions
- Evaluation pipeline using an LLM-as-judge to score faithfulness and answer relevance, including automatic hallucination detection
- Guardrails: input validation, a maximum-step limit to prevent runaway agent loops, and output checks
- Configurable, model-agnostic design (the Qdrant URL and LLM are set through environment variables)
- Containerized with Docker and deployed live on Railway

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Vector database | Qdrant |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) |
| Retrieval | Dense vectors, BM25, Reciprocal Rank Fusion, cross-encoder re-ranking |
| Agent framework | LangGraph |
| LLM | Groq (openai/gpt-oss-120b) |
| Backend | FastAPI |
| Frontend | HTML/JS chat interface |
| Containerization | Docker |
| Deployment | Railway |

## Project structure

```
agentic-rag-assistant/
├── ingestion/        Chunking and embedding experiments
├── rag/              Retrieval: vector search, hybrid search, re-ranking, full RAG pipeline
├── agent/            Tool calling, LangGraph agent, evaluation, guardrails
├── shared/           Reusable production agent used by the API
├── api/              FastAPI backend and chat frontend
├── Dockerfile        Container build for deployment
└── requirements.txt
```

## Running locally

Prerequisites: Python 3.13, Docker.

```bash
# 1. Clone and enter the project
git clone https://github.com/Atharva-ml-sys/agentic-rag-assistant.git
cd agentic-rag-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant (vector database)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 5. Add your Groq API key to a .env file
echo "GROQ_API_KEY=your_key_here" > .env

# 6. Run the API
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

## Example

Request:

```bash
curl -X POST https://agentic-rag-assistant-production.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much does CloudDesk cost?"}'
```

Response:

```json
{
  "question": "How much does CloudDesk cost?",
  "answer": "CloudDesk is priced at 499 rupees per user per month."
}
```

## Notes and roadmap

The current knowledge base uses a small sample dataset to demonstrate the pipeline end to end. Planned improvements include expanding the indexed documents, wiring the hybrid-search and re-ranking stages directly into the deployed agent, tightening the agent's grounding on out-of-scope queries, adding cost and latency tracing, and a CI evaluation gate that blocks merges when answer quality regresses.
