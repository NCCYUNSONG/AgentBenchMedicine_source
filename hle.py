#!/usr/bin/env python3

import os
import json
import time
import asyncio
import configparser
import openai
from app.agent.manus_backup import Manus

# --- Load config and set up Llama4 Maverick and GPT-4o-Mini clients ---
def load_config():
    config = configparser.ConfigParser()
    with open("config.ini", "r", encoding="utf-8") as f:
        config.read_file(f)
    return config

config = load_config()

llama_client = openai.OpenAI(
    base_url=config["llama4_maverick"]["base_url"],
    api_key=config["llama4_maverick"]["api_key"],
)
llama_model_name = config["llama4_maverick"]["model"]

# --- Paths ---
INPUT_FILE  = './hle/HLE.json'
OUTPUT_FILE = 'openmanus_results.json'

# --- Helpers ---
def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def _run_manus(prompt: str):
    agent = Manus()
    return await agent.run(prompt)

def run_manus(prompt: str):
    return asyncio.run(_run_manus(prompt))

def process_sample(sample: dict, time_offset: float, local_start: float) -> dict:
    question = sample.get("question", "").strip()
    prompt = question

    # 1) Run Manus agent
    manus_resp = run_manus(prompt)
    predicted = getattr(manus_resp, "answer", str(manus_resp)).strip()

    # 2) Compute cumulative time_elapsed
    elapsed = time_offset + (time.time() - local_start)

    return {
        "id":               sample.get("id"),
        "question":         question,
        "answer":           sample.get("answer", "").strip(),
        "predicted_answer": predicted,
        "time_elapsed":     elapsed,
    }

def main():
    samples = load_json(INPUT_FILE)

    # --- Load or initialize results, determine time offset ---
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and "run_started_at" in data:
            run_started_at = data["run_started_at"]
            results       = data.get("results", [])
            time_offset   = 0.0
            print(f"Resuming (new format) from {len(results)} existing results.")
        else:
            results = data if isinstance(data, list) else []
            time_offset = max((r.get("time_elapsed", 0.0) for r in results), default=0.0)
            run_started_at = time.time()
            print(f"Resuming (legacy format) from {len(results)} existing results; "
                  f"time_offset={time_offset:.2f}s")
    else:
        results       = []
        time_offset   = 0.0
        run_started_at = time.time()
        print("Starting fresh run.")

    processed_ids = {r["id"] for r in results if "id" in r}
    local_start   = run_started_at

    # --- Process remaining samples ---
    for sample in samples:
        sid = sample.get("id")
        if sid in processed_ids:
            continue

        res = process_sample(sample, time_offset, local_start)
        results.append(res)
        processed_ids.add(sid)

        out = {
            "run_started_at": run_started_at,
            "results": results
        }
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

    total = len(results)
    print(f"Completed {total} samples.")

if __name__ == "__main__":
    main()
