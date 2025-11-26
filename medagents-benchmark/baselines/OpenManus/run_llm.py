import os
import json
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import punctuation
import requests  # Replacing openai with requests for local API calls

load_dotenv()

# Configuration for your local LLaMA 4 server
import configparser
import openai

# Load configuration from config.ini
def load_config():
    config = configparser.ConfigParser()
    with open("config.ini", "r") as f:
        config.read_file(f)
    return config

config = load_config()

# Set up OpenAI client for llama4_maverick using OpenAI-compatible client
llama_client = openai.OpenAI(
    base_url=config["llm"]["base_url"],
    api_key=config["llm"]["api_key"]
)
llama_model_name = config["llm"]["model"]




def save_results(results, output_file):
    results = sorted(results, key=lambda x: x['id'])
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def run_llama_4(prompt: str) -> tuple[str, dict]:
    """
    Calls local LLaMA 4 using OpenAI-style client loaded from config.ini.
    Returns predicted answer and token usage.
    """
    system_prompt = (
        "You are a QA assistant.\n"
        "Given the following multiple-choice question and options, "
        "choose the most appropriate answer and return only a single letter, nothing else.\n"
        "DO NOT explain your answer or generate steps."
    )

    try:
        response = llama_client.chat.completions.create(
            model=llama_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        }
        return answer, usage
    except Exception as e:
        print(f"LLaMA 4 API error: {e}")
        return "N/A", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}



def process_sample(idx, raw_sample, args):
    try:
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        options = raw_sample['options']
        gold_answer = raw_sample['answer_idx']

        if isinstance(options, dict):
            ordered_keys = sorted(options.keys())
            formatted_options = "\n".join([f"{key}. {options[key]}" for key in ordered_keys])
            gold_answer = raw_sample['answer_idx']
        else:
            formatted_options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
            gold_answer = raw_sample['answer_idx']

        prompt = f"Question: {question.strip()}\nOptions:\n{formatted_options}\n\nPlease choose the most appropriate answer."

        print(f"\n========== Prompt for Sample {idx} ==========")
        print(prompt)
        print("===========================================\n")

        # Time specifically the model call
        start = time.time()
        predicted, usage = run_llama_4(prompt)
        elapsed = time.time() - start

        if isinstance(gold_answer, int):
            gold_answer = chr(65 + gold_answer)

        return {
            "id": raw_sample.get("id", idx),
            "realidx": raw_sample.get("realidx", idx),
            "question": question.strip(),
            "options": options,
            "answer": gold_answer,
            "predicted_answer": predicted,
            "time_elapsed": elapsed,
            "token_usage": usage
        }

    except Exception as e:
        print(f"[ERROR] Failed on sample {raw_sample.get('id', '?')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gemma')  # Updated to reflect the local model
    parser.add_argument('--dataset_name', default='medmcqa')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--split', default='test_hard')
    parser.add_argument('--output_files_folder')
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    args = parser.parse_args()

    args.start_time = time.time()

    # Verify API URL

    input_file = os.path.join(args.dataset_dir, f"{args.split}.jsonl")
    samples = load_jsonl(input_file)

    if 'realidx' not in samples[0]:
        samples = [{**s, 'realidx': idx} for idx, s in enumerate(samples)]

    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)

    output_file = os.path.join(subfolder, f"{args.model_name}-{args.dataset_name}-{args.split}.json")
    print(f"Saving results to: {output_file}")

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    processed_ids = {r['id'] for r in results}

    end_pos = len(samples) if args.end_pos == -1 else args.end_pos
    test_range = [i for i in range(args.start_pos, end_pos) if samples[i].get("id") not in processed_ids]

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = [
            executor.submit(process_sample, idx, samples[idx], args)
            for idx in tqdm(test_range, desc="Processing")
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            result = future.result()
            if result:
                results.append(result)
                try:
                    save_results(results, output_file)
                except Exception as e:
                    print(f"Failed to save results: {e}")
                    break  # Exit loop on save failure

    print(f"Saved {len(results)} results to {output_file}")
    executor.shutdown(wait=True)
    print("Program exiting")


if __name__ == "__main__":
    main()
