#!/usr/bin/env python3

import os
import json
import time
import logging
import configparser
from typing import Dict, List, Any

import pandas as pd
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Global token counter (only for Llama calls) ---
token_usage_counter = {
    "llama4_maverick": 0
}

# --- Load config and set up Llama4 Maverick client ---
def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    with open("config.ini", "r", encoding="utf-8") as f:
        cfg.read_file(f)
    return cfg

config = load_config()

llama_client = openai.OpenAI(
    base_url=config["llama4_maverick"]["base_url"],
    api_key=config["llama4_maverick"]["api_key"]
)
llama_model = config["llama4_maverick"]["model"]

# --- Paths ---
INPUT_FILE  = "./hle/bio_med_with_image.parquet"
OUTPUT_FILE = "llama4_maverick_image_results.json"

# --- Helpers ---
def load_parquet(path: str) -> List[Dict[str, Any]]:
    """Load Parquet and return list of samples."""
    df = pd.read_parquet(path)
    required = {"id", "question", "image", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    records = df.to_dict(orient="records")
    logging.info(f"Loaded {len(records)} samples from '{path}'")
    return records

def ask_llama_with_image(prompt: str, image_b64: str) -> Dict[str, Any]:
    """
    Send prompt + image to Llama4 and return answer text & token usage.
    Falls back to word‐count if resp.usage is None.
    Updates global llama token counter.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}}
            ]
        }
    ]
    resp = llama_client.chat.completions.create(
        model=llama_model,
        messages=messages,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content.strip()

    if resp.usage:
        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        tt = resp.usage.total_tokens
    else:
        pt = len(prompt.split())
        ct = len(answer.split())
        tt = pt + ct

    token_usage_counter["llama4_maverick"] += tt
    logging.info(f"[llama4_maverick] prompt={pt}, completion={ct}, total={tt}")

    return {"answer": answer, "tokens": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}}

def process_sample(sample: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Query Llama4 with image and record response and timing."""
    q = sample["question"].strip()
    img = sample["image"]
    correct = sample["answer"].strip()

    llama_resp = ask_llama_with_image(q, img)
    predicted = llama_resp["answer"]
    llama_tokens = llama_resp["tokens"]

    elapsed = time.time() - start_time

    return {
        "id": sample.get("id"),
        "question": q,
        "answer": correct,
        "predicted_answer": predicted,
        "time_elapsed": elapsed,
        "llama_tokens": llama_tokens
    }

def main():
    samples = load_parquet(INPUT_FILE)
    start_time = time.time()

    # Load or initialize results
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            logging.info(f"Resuming with {len(results)} existing results.")
        else:
            results = data if isinstance(data, list) else []
            logging.info(f"Resuming legacy format with {len(results)} results.")
    else:
        results = []
        logging.info("Starting fresh run.")

    processed_ids = {r["id"] for r in results if "id" in r}

    # Process each sample
    for sample in samples:
        sid = sample.get("id")
        if sid in processed_ids:
            continue

        try:
            res = process_sample(sample, start_time)
        except Exception as e:
            logging.error(f"Error processing sample ID {sid}: {e}")
            continue

        results.append(res)
        processed_ids.add(sid)

        # Save incrementally
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"Processed {len(results)} samples.")
    print(f"Total Llama4 tokens consumed: {token_usage_counter['llama4_maverick']}")

if __name__ == "__main__":
    main()
