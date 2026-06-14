"""
Chunking Demo - dekhte hain ki ek document chote tukdo (chunks) mein
kaise tota jaata hai. Yeh RAG ka pehla aur sabse important step hai.
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# --- Yeh hamara sample document hai (jaise koi company ka doc ho) ---
sample_text = """Agentic RAG systems combine retrieval and reasoning. 
A retrieval-augmented generation pipeline first finds relevant documents. 
Then it passes them to a language model to generate grounded answers. 

The key advantage is reduced hallucination. The model answers from real data, 
not just its memory. This makes the system trustworthy for production use.

Chunking is the first step. Documents are split into smaller pieces. 
Each piece is later converted into a vector embedding for search. 
Good chunking preserves meaning and improves retrieval quality."""

print("=" * 70)
print("ORIGINAL TEXT LENGTH:", len(sample_text), "characters")
print("=" * 70)

# --- TAREEKA 1: Fixed-size chunking (simple, par meaning tod sakta hai) ---
print("\n\n>>> METHOD 1: CharacterTextSplitter (fixed-size)\n")

fixed_splitter = CharacterTextSplitter(
    separator="",        # kisi bhi character pe kaat do
    chunk_size=150,      # har chunk approx 150 characters ka
    chunk_overlap=20,    # har chunk thoda overlap karega (context na toote)
)
fixed_chunks = fixed_splitter.split_text(sample_text)

for i, chunk in enumerate(fixed_chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()

# --- TAREEKA 2: Recursive chunking (smart, meaning preserve karta hai) ---
print("\n\n>>> METHOD 2: RecursiveCharacterTextSplitter (smart)\n")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    # Yeh pehle paragraph(\n\n) pe todega, phir line(\n), phir space pe
    separators=["\n\n", "\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(sample_text)

for i, chunk in enumerate(recursive_chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()

print("=" * 70)
print(f"Fixed-size ne banaye: {len(fixed_chunks)} chunks")
print(f"Recursive ne banaye:  {len(recursive_chunks)} chunks")
print("=" * 70)