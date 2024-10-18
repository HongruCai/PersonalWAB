import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import sys
from utils import retrieve_top_k_memories, build_taskspe_memory, prettify_product_info, PARAM_PROMPT

# Tokenizers and Model loading
llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')


def get_chosen_reject(options):
    max_score = max(options.values())
    min_score = min(options.values())
    chosen = None
    reject = None
    for key, score in options.items():
        if score == max_score and chosen is None:
            chosen = key  # First max score encountered
        if score == min_score:
            reject = key
    return chosen, reject


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare param generation data for DPO')
    parser.add_argument('--task_file', type=str, required=True, help='Path to the task file')
    parser.add_argument('--user_history_file', type=str, required=True, help='Path to the user history file')
    parser.add_argument('--dpo_data_file', type=str, required=True, help='Path to DPO data file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output results')
    parser.add_argument('--mem_token_length', type=int, default=768, help='Maximum token length of memory sequence')
    parser.add_argument('--mem_length', type=int, default=100, help='Maximum length of memory sequence')
    return parser.parse_args()


def main():
    args = parse_args()

    task_file = json.load(open(args.task_file, 'r'))
    user_history = json.load(open(args.user_history_file, 'r'))
    dpo_data = json.load(open(args.dpo_data_file, 'r'))
    
    llama_data = {'train': [], 'test': []}

    for split, tasks in task_file.items():
        for task in tqdm(tasks):
            input_text = task['task']
            user_id = task['user_id']
            task_type = task['type']
            timestamp = task['timestamp']
            product_info = task['target']['product_info']

            history = [item for item in user_history[user_id] if item['review']["timestamp"] < timestamp]
            history = retrieve_top_k_memories(input_text, history, sim_model, sim_tokenizer, k=args.mem_length) if history else []

            chosen, rej = get_chosen_reject(dpo_data[input_text])

            mem = build_taskspe_memory(history, task_type)
            prefix_text = PARAM_PROMPT.replace('<Instruction>', input_text + (prettify_product_info(product_info) if task_type == 'review' else ''))
            memory_text = '|'.join(mem)
            tool_text = 'search_product_by_query' if task_type == 'search' else ('get_recommendations_by_history' if task_type == 'recommend' else 'add_product_review')

            tokenized_memory = llama_tokenizer(memory_text, return_tensors=None, truncation=True, max_length=args.mem_token_length)
            memory_text_truncated = llama_tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
            truncated_full_text = prefix_text.replace('<Memory>', memory_text_truncated).replace('<Tool>', tool_text)

            llama_data[split].append({'instruction':input_text, 'prompt': truncated_full_text, 'chosen': chosen, 'rejected': rej})

    with open(args.output_file, 'w') as f:
        json.dump(llama_data, f, indent=2)

if __name__ == '__main__':
    main()
