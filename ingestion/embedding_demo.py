"""
Embedding Demo - text ko numbers (vectors) mein badalna aur dekhna ki
similar meaning wale texts ke vectors kitne 'paas' hote hain.
Yeh semantic search ki neev hai.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Model load karo (pehli baar ~90MB download hoga, phir cache se chalega)
print("Model load ho raha hai... (pehli baar thoda time lagega)\n")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready!\n")

# --- Kuch sentences lete hain ---
sentences = [
    "The dog is playing in the park.",      # 0
    "A puppy runs around the garden.",      # 1  (meaning: #0 jaisa)
    "I need to buy a new car.",             # 2  (bilkul alag topic)
]

# Har sentence ko embedding (vector) mein badlo
embeddings = model.encode(sentences)

# Dekhte hain ek embedding dikhta kaisa hai
print("Pehle sentence ka embedding (sirf pehle 8 numbers dikha rahe):")
print(embeddings[0][:8], "...")
print(f"\nPoore embedding ki length: {len(embeddings[0])} numbers")
print("(matlab har sentence 384-dimension ke vector mein badal gaya)\n")
print("=" * 60)

# --- Ab similarity check karte hain ---
# cos_sim = do vectors kitne 'paas' hain (1.0 = bilkul same, 0 = unrelated)
print("\nSIMILARITY SCORES (1.0 = bilkul similar, 0 = unrelated):\n")

sim_dog_puppy = cos_sim(embeddings[0], embeddings[1]).item()
sim_dog_car = cos_sim(embeddings[0], embeddings[2]).item()

print(f"'dog playing'  vs  'puppy running'  =  {sim_dog_puppy:.3f}   <- similar meaning")
print(f"'dog playing'  vs  'buy a car'      =  {sim_dog_car:.3f}   <- alag topic")
print()
print("Dekh: dog/puppy ka score zyada hai (paas), dog/car ka kam (door).")
print("Computer ne 'meaning' pakda - bina koi word match kiye!")