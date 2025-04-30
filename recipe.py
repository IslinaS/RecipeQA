import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import google.generativeai as genai
from bert_score import score
import pickle
from CLIP import embed_query_text_to_vector

os.environ["GOOGLE_API_KEY"]="secret/api/key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

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

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast
# retrieve saved index
index = faiss.read_index("recipe_index.faiss")
print(f"retrieved index")

# Load FAISS index
image_index = faiss.read_index("image_index.faiss")
print(f"loaded image faiss index")

# Load filenames list
with open("image_filenames.pkl", "rb") as f:
    image_filenames = pickle.load(f)
print(f"loaded image filenames")

# Retrieve top k related recipes
def retrieve_top_k(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    print(f"retrieved top k docs successfully")
    return [docs[i] for i in I[0]]

# Retrieve top k related images
def retrieve_top_k_images(query_img, k=5):
    query_vector = embed_query_text_to_vector(query_img)
    D, I = image_index.search(query_vector, k)
    retrieved = [image_filenames[i] for i in I[0]]
    print(f"retrieved image file names: ", retrieved)
    return retrieved

def build_context_from_images(retrieved, image_dir):
    context_lines = []
    for fname in retrieved:
        path = os.path.join(image_dir, fname)
        context_lines.append(f"Image: {fname}, Path: {path}")
    return "\n".join(context_lines)

# LLM response generation: gemini 2.0 flash
def generate_with_rag_gemini(query, k=5, context=True, image=True, image_dir=None):
    image_context=""
    image_retrieved=""
    context=""
    retrieved=""

    if context:
        retrieved = retrieve_top_k(query, k=k)
        context = "\n\n".join([f"{doc['title']}: {doc['text']}" for doc in retrieved])
        if image and image_dir is not None:
            image_retrieved = retrieve_top_k_images(query, k=k)
            image_context = build_context_from_images(image_retrieved, image_dir)
    prompt = f"""You are a helpful and creative AI chef assistant. Here are some recipes from previous documents, including ingredients, steps, and cooking directions: {context}. Here are some relevant images to those recipes: {image_context}
    Now based on this context, generate a recipe including name of the recipe, ingredients, and cooking instructions in response to: "{query}"
    """

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    print(f"generating response...")
    response = model.generate_content(prompt)

    return response.text, retrieved, image_retrieved

# LLM: GPT2
def generate_with_rag_gpt2(query, k=1, context=True, image=True, image_dir=None):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    image_context=""
    image_retrieved=""
    context=""
    retrieved=""

    if context:
        retrieved = retrieve_top_k(query, k=k)
        context = "\n\n".join([f"{doc['title']}: {doc['text']}" for doc in retrieved])
        if image and image_dir is not None:
            image_retrieved = retrieve_top_k_images(query, k=k)
            image_context = build_context_from_images(image_retrieved, image_dir)
    prompt = f"""You are a helpful and creative AI chef assistant. Here are some recipes from previous documents, including ingredients, steps, and cooking directions: {context}. Here are some relevant images to those recipes: {image_context}
    Now based on this context, generate a recipe including name of the recipe, ingredients, and cooking instructions in response to: "{query}"
    """
    print(f"generating response...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.7)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text, retrieved, image_retrieved


# Evaluate response against test set
def bertscore_evaluate(references, candidates, lang="en"):
    # Each should be a list of strings (ground_truth, generated)
    P, R, F1 = score(candidates, references, lang=lang)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="gemini or GPT2")
    parser.add_argument("--query", type=str, required=True,
                        help="Query from the user to generate a certain type of recipe.")
    parser.add_argument("--k", type=int, 
                        help="number of top related documents from RecipeQA")
    parser.add_argument("--context", action="store_true",
                        help="Specifies whether retrieved context from RAG is considered")
    parser.add_argument("--image", action="store_true", 
                        help="Whether to include images in the context")
    parser.add_argument("--image_dir", type=str,
                        help="directory to the images")
    args = parser.parse_args()
    if args.model == "gemini":
        response, retrieved, image_retrieved = generate_with_rag_gemini(args.query, args.k, args.context, args.image, args.image_dir)
    elif args.model == "GPT2":
        response, retrieved, image_retrieved = generate_with_rag_gpt2(args.query, args.k, args.context, args.image, args.image_dir)

    # return top k recipe ids
    recipe_ids = []
    if args.context == True:
        for r in retrieved:
            recipe_ids.append(r.get("recipe_id"))
    print(f"Here is the generated recipe: {response} based on the following top {args.k} contexts: {recipe_ids} and the top {args.k} images: {image_retrieved}")
    # test

    references = []
    generated = []

    for doc in test_docs[:10]:  # limit for cost
        query = doc["title"]
        ground_truth = doc["text"]

        if args.model == "gemini":
            gen_text, doc_retrieved, image_retrieved = generate_with_rag_gemini(query, k=5)
        if args.model == "GPT2":
            gen_text, doc_retrieved, image_retrieved = generate_with_rag_gpt2(query, k=1)

        references.append(ground_truth)
        generated.append(gen_text)

    # Compute BERTScore
    results = bertscore_evaluate(references, generated)
    print("BERTScore on test set:", results)
