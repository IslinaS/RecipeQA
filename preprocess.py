import os
import json
import argparse
import pandas as pd
from tqdm import tqdm


def load_recipeqa_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"data type: ", type(data))
    print(f"first key: ", next(iter(data)))
    return data


def extract_context_entries(data, split_name):
    entries = []

    for items in tqdm(data.values(), desc=f"Processing {split_name}"):
        for item in items:
            recipe_id = item.get("recipe_id", "unknown_recipe")
            context_steps = item.get("context", [])

            for step in context_steps:
                step_id = step.get("id", None)
                if step_id is None:
                    continue

                step_title = step.get("title", "").strip()
                step_body = step.get("body", "").strip()

                if not step_body:
                    continue

                entry_id = f"{recipe_id}_step_{step_id}"
                full_text = f"{step_title}: {step_body}"

                entries.append({
                    "id": entry_id,
                    "recipe_id": recipe_id,
                    "step_num": step_id,
                    "title": step_title,
                    "text": full_text,
                    "split": split_name
                })

    return entries


def preprocess_all_splits(data_dir, output_path):
    all_entries = []

    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping...")
            continue

        data = load_recipeqa_file(file_path)
        entries = extract_context_entries(data, split)
        all_entries.extend(entries)

    df = pd.DataFrame(all_entries)
    df.to_json(output_path, orient="records", lines=True)
    print(f"✅ Preprocessed context data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with train/val/test JSON files from RecipeQA.")
    parser.add_argument("--output", type=str, default="recipeqa_dense_docs.jsonl",
                        help="Output path for preprocessed JSONL file.")
    args = parser.parse_args()

    print(f"retrieved args")
    preprocess_all_splits(args.data_dir, args.output)