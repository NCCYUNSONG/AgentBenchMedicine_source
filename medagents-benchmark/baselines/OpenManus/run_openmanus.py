import os
import json
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.agent.manus import Manus
from app.logger import logger
from string import punctuation

load_dotenv()

def save_results(results, output_file):
    results = sorted(results, key=lambda x: x['id'])
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def run_agent_on_sample(prompt: str):
    import asyncio
    async def _run():
        agent = Manus()
        return await agent.run(prompt)
    return asyncio.run(_run())

def process_sample(idx, raw_sample, args):
    try:
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        options = raw_sample['options']
        gold_answer = raw_sample['answer_idx']

        if isinstance(options, dict):
            # Ensure consistent order A-D
            ordered_keys = sorted(options.keys())
            formatted_options = "\n".join([f"{key}. {options[key]}" for key in ordered_keys])
            gold_answer = raw_sample['answer_idx']  # Should be like 'A'
        else:
            formatted_options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
            gold_answer = raw_sample['answer_idx']  # Should be like 0
        prompt = f"Question: {question.strip()}\nOptions:\n{formatted_options}\n\nPlease choose the most appropriate answer."

        print(f"\n========== Prompt for Sample {idx} ==========")
        print(prompt)
        print("===========================================\n")

        predicted = run_agent_on_sample(prompt)

        return {
            "id": raw_sample.get("id", idx),
            "realidx": raw_sample.get("realidx", idx),
            "question": question.strip(),
            "options": options,
            "answer": gold_answer,
            "predicted_answer": predicted,
            "time_elapsed": time.time() - args.start_time,
        }
    except Exception as e:
        logger.warning(f"[ERROR] Failed on sample {raw_sample.get('id', '?')}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='openmanus')
    parser.add_argument('--dataset_name', default='medexqa')
    parser.add_argument('--dataset_dir', default='../../data/medexqa/')
    parser.add_argument('--split', default='test_hard')
    parser.add_argument('--output_files_folder', default='./output/')
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    args = parser.parse_args()

    args.start_time = time.time()

    input_file = os.path.join(args.dataset_dir, f"{args.split}.jsonl")
    samples = load_jsonl(input_file)

    if 'realidx' not in samples[0]:
        samples = [{**s, 'realidx': idx} for idx, s in enumerate(samples)]

    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)

    output_file = os.path.join(subfolder, f"{args.model_name}-{args.dataset_name}-{args.split}.json")
    logger.info(f"Saving results to: {output_file}")

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results")

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
                    logger.error(f"Failed to save results: {e}")
                    break  # Exit loop on save failure

    print(f"Saved {len(results)} results to {output_file}")
    executor.shutdown(wait=True)
    print("Program exiting")
if __name__ == "__main__":
    main()
