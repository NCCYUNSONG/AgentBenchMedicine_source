#!/usr/bin/env python3

import json
import time
import configparser
import openai
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Load config and set up Llama4 client ---
def load_config():
    config = configparser.ConfigParser()
    with open("config.ini", "r", encoding="utf-8") as f:
        config.read_file(f)
    return config

config = load_config()
llama_client = openai.OpenAI(
    base_url=config["gpt-4.1"]["base_url"],
    api_key=config["gpt-4.1"]["api_key"]
)
llama_model = config["gpt-4.1"]["model"]

# --- Paths ---
INPUT_FILE = './hle/HLE.json'
OUTPUT_FILE = 'gpt-4.1_results_no_judge.json'

# --- Helpers ---
def load_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} records from '{path}'")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON file '{path}': {e}")
        raise

def ask_llama(question: str) -> Dict[str, Any]:
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant that answers questions."
    }
    user_msg = {
        "role": "user",
        "content": question
    }
    try:
        resp = llama_client.chat.completions.create(
            model=llama_model,
            messages=[system_msg, user_msg],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        tokens = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0
        }
        logging.info(f"Tokens: prompt={tokens['prompt_tokens']}, "
                     f"completion={tokens['completion_tokens']}, total={tokens['total_tokens']}")
        return {"answer": answer, "tokens": tokens}
    except Exception as e:
        logging.error(f"Error in ask_llama: {e}")
        raise

def process_sample(sample: dict, start_time: float) -> Dict[str, Any]:
    question = sample.get("question", "").strip()
    options = sample.get("options")
    if options:
        if isinstance(options, dict):
            opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        else:
            opts_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        prompt = f"{question}\n\nOptions:\n{opts_text}\n\nPlease choose the correct answer."
    else:
        prompt = question

    try:
        llama_response = ask_llama(prompt)
        predicted = llama_response["answer"]
        tokens = llama_response["tokens"]
    except Exception as e:
        logging.error(f"Failed to process sample ID {sample.get('id', 'unknown')}: {e}")
        predicted = ""
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "id": sample.get("id"),
        "question": question,
        "answer": sample.get("answer", "").strip(),
        "predicted_answer": predicted,
        "time_elapsed": time.time() - start_time,
        "tokens": tokens
    }

def main():
    try:
        samples = load_json(INPUT_FILE)
        results = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        start_time = time.time()

        for sample in samples:
            res = process_sample(sample, start_time)
            results.append(res)
            for key in total_tokens:
                total_tokens[key] += res["tokens"][key]

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Processed {len(results)} samples.")
        print("Total Token Consumption:")
        print(f"  Prompt Tokens:     {total_tokens['prompt_tokens']}")
        print(f"  Completion Tokens: {total_tokens['completion_tokens']}")
        print(f"  Total Tokens:      {total_tokens['total_tokens']}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
