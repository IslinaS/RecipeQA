# /usr/xtmp/ys298/RecipeQA/images-qa/train/images-qa

import torch
import clip
from PIL import Image
import os
import faiss
import pickle
import numpy as np
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def build_image_index(image_dir, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = []
    filenames = []
    features = []

    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in all_files:
        path = os.path.join(image_dir, filename)
        image = preprocess(Image.open(path)).unsqueeze(0)
        images.append(image)
        filenames.append(filename)

        if len(images) == batch_size:
            batch = torch.cat(images, dim=0).to(device)
            with torch.no_grad():
                batch_features = model.encode_image(batch)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            features.append(batch_features.cpu().numpy())
            images = []

    # Process any leftover images
    if images:
        batch = torch.cat(images, dim=0).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu().numpy())

    features = np.vstack(features)

    index = faiss.IndexFlatIP(features.shape[1])  # Inner product = cosine similarity if normalized
    index.add(features)

    return index, filenames


def embed_query(query):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def embed_query_text_to_vector(query_text):
    text = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory to the images")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    index, filenames = build_image_index(args.image_dir, args.batch_size)
    # Save FAISS index
    faiss.write_index(index, "image_index.faiss")

    # Save filenames list
    with open("image_filenames.pkl", "wb") as f:
        pickle.dump(filenames, f)
