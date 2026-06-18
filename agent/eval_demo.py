"""
Evaluation Demo - measures the quality of our RAG answers using the
"LLM-as-judge" technique.

We score two things on a 1-5 scale:
  - Faithfulness: is the answer grounded in the retrieved context,
                  or did the model make something up (hallucination)?
  - Relevance:    does the answer actually address the question?

A separate LLM acts as the judge, at temperature 0 for consistent scoring.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ----------------------------------------------------------------------
# Set up the knowledge base (same docs as our agent)
# ----------------------------------------------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "eval_kb"

knowledge = [
    "CloudDesk costs 499 rupees per user per month.",
    "CloudDesk offers a 30-day free trial with no credit card required.",
    "TechCorp was founded in 2015 in Bangalore, India.",
    "Customer support is available Monday to Friday, 9 AM to 6 PM IST.",
    "To reset your password, go to Settings and click 'Reset Password'.",
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
# A simple RAG function (retrieve + generate) - this is what we evaluate
# ----------------------------------------------------------------------
def rag_answer(question):
    qv = embed_model.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION, query=qv, limit=3
    ).points
    context = "\n".join(r.payload["text"] for r in results)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return context, response.choices[0].message.content


# ----------------------------------------------------------------------
# THE JUDGE: a second LLM call that scores the answer 1-5
# ----------------------------------------------------------------------
def judge(question, context, answer):
    judge_prompt = f"""You are a strict evaluator of an AI assistant's answer.

Given the QUESTION, the CONTEXT the assistant was given, and its ANSWER,
score the answer on two criteria from 1 (worst) to 5 (best):

1. faithfulness: Is every claim in the ANSWER supported by the CONTEXT?
   (5 = fully grounded, 1 = makes up facts not in context)
2. relevance: Does the ANSWER actually address the QUESTION?
   (5 = directly answers, 1 = off-topic)

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Respond ONLY with valid JSON in this exact format, nothing else:
{{"faithfulness": <int>, "relevance": <int>, "reason": "<short reason>"}}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,  # judge must be consistent
    )
    raw = response.choices[0].message.content.strip()
    # strip markdown fences if the model added them
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ----------------------------------------------------------------------
# EVAL DATASET: questions to test the system on
# (mix of answerable questions and one that's NOT in the docs)
# ----------------------------------------------------------------------
eval_questions = [
    "How much does CloudDesk cost?",
    "Is there a free trial?",
    "When was TechCorp founded?",
    "Who is the CEO of TechCorp?",        # NOT in docs - should say "I don't know"
]

# ----------------------------------------------------------------------
# RUN THE EVAL
# ----------------------------------------------------------------------
print("=" * 70)
print("RUNNING EVALUATION\n")

faith_scores = []
rel_scores = []

for q in eval_questions:
    context, answer = rag_answer(q)
    scores = judge(q, context, answer)

    faith_scores.append(scores["faithfulness"])
    rel_scores.append(scores["relevance"])

    print(f"Q: {q}")
    print(f"A: {answer}")
    print(f"   -> faithfulness: {scores['faithfulness']}/5 | "
          f"relevance: {scores['relevance']}/5")
    print(f"   -> judge says: {scores['reason']}\n")

# ----------------------------------------------------------------------
# OVERALL SCORES
# ----------------------------------------------------------------------
avg_faith = sum(faith_scores) / len(faith_scores)
avg_rel = sum(rel_scores) / len(rel_scores)

print("=" * 70)
print(f"AVERAGE FAITHFULNESS: {avg_faith:.2f}/5")
print(f"AVERAGE RELEVANCE:    {avg_rel:.2f}/5")
print("=" * 70)