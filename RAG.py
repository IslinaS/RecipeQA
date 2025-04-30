import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"]="secret/api/key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load data
docs = []
with open("recipeqa_dense_docs.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

# Get texts from training split to embed
train_docs = [doc for doc in docs if doc["split"] == "train"]
texts = [doc["text"] for doc in train_docs]
ids = [doc["id"] for doc in train_docs]

val_docs = [d for d in docs if d["split"] == "val"]
val_texts = [doc["text"] for doc in val_docs]
val_ids = [doc["id"] for doc in val_docs]

test_docs = [d for d in docs if d["split"] == "test"]
test_texts = [doc["text"] for doc in test_docs]
test_ids = [doc["id"] for doc in test_docs]

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, "recipe_index.faiss")

