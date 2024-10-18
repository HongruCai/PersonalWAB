import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import sys
from utils import retrieve_top_k_memories, build_taskspe_memory, prettify_product_info, generate_search_query, PARAM_PROMPT, FUNCTION_PROMPT


llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare param generation data for SFT')
    parser.add_argument('--task_file', type=str, required=True, help='Path to the task file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output results')
    return parser.parse_args()


def main():
    args = parse_args()

    task_file = json.load(open(args.task_file, 'r'))
    llama_data = {'train': [], 'test': []}

    for split, tasks in task_file.items():
        for task in tqdm(tasks):
            input_text = task['task']
            task_type = task['type']

            tool_text = 'search_product_by_query' if task_type == 'search' else ('get_recommendations_by_history' if task_type == 'recommend' else 'add_product_review')

            full_text = FUNCTION_PROMPT.replace('<Instruction>', input_text)
            llama_data[split].append({'instruction':input_text,'prompt': full_text, 'target': tool_text})

    with open(args.output_file, 'w') as f:
        json.dump(llama_data, f, indent=2)

if __name__ == '__main__':
    main()
