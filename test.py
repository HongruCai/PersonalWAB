import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
from PersonalWAB.envs.pwab.functions.get_recommendations_by_history import get_recommendations_by_history
from PersonalWAB.envs.pwab.functions.search_product_by_query import search_product_by_query
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PWA tasks")
    parser.add_argument('--evaluate_dpo', type=str, default='False', help='Whether to evaluate DPO')
    parser.add_argument('--task_file', type=str, default='PersonalWAB/envs/pwa/data/user_instructions.json', help='Path to task file')
    parser.add_argument('--param_file', type=str, default='PUMA/output/res/', help='Path to tool input file')
    parser.add_argument('--function_file', type=str, default='PUMA/output/', help='Path to tool selected file')
    parser.add_argument('--all_products', type=str, default='PersonalWAB/envs/pwa/data/all_products.json', help='Path to all products file')
    parser.add_argument('--dpo_output', type=str, default='PUMA/data/dpo_data.json', help='Path to DPO output file')
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model.to('cuda')

def compute_similarity(target_review, agent_review):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input.to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    del model_output
    del sentence_embeddings
    torch.cuda.empty_cache()
    return similarity

def trucate_text(text, max_length):
    tokenized_memory = llama_tokenizer(text, return_tensors=None, truncation=True, max_length=max_length)
    truncated_memory_ids = tokenized_memory["input_ids"]
    memory_text_truncated = llama_tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
    return memory_text_truncated

args = parse_args()

tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))

final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}

llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

if args.evaluate_dpo == 'True':
    all_res = {}
    for task in tqdm(tasks['train']):
        task_type = task['type']
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        cur_res = {}
        if task_type == 'search':
            query = tool_input[instructions]
            for q in query:
                res = search_product_by_query(data={}, query=q)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break      
                cur_res[q] = score
        elif task_type == 'recommend':
            history = tool_input[instructions]
            for h in history:
                h_ = [item.strip() for item in h.split(',')]
                h_ = list(set(h_))
                res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break
                cur_res[h] = score
        else:
            review = tool_input[instructions]
            for r in review:
                target_review = task['target']['review']['text']
                agent_review = r
                similarity = compute_similarity(target_review, agent_review)
                cur_res[r] = similarity
        all_res[instructions] = cur_res
    with open(args.dpo_output, 'w') as f:
        json.dump(all_res, f, indent=2)
else:
    for task in tqdm(tasks['test']):
        tool = tool_selected[task['task']][0]
        if tool == 'search_product_by_query':
            task_type = 'search'
        elif tool == 'get_recommendations_by_history':
            task_type = 'recommend'
        else:
            task_type = 'review'
        gt_task_type = task['type']
        if task_type == gt_task_type:
            tool_accuracy[gt_task_type].append(1)
        else:
            tool_accuracy[gt_task_type].append(0)
            final_results[gt_task_type].append(0)
            continue
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        if task_type == 'search':
            query = tool_input[instructions]
            for q in query:
                res = search_product_by_query(data={}, query=q)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break      
            final_results['search'].append(score)
        elif task_type == 'recommend':
            history = tool_input[instructions]
            for h in history:
                h_ = [item.strip() for item in h.split(',')]
                h_ = list(set(h_))
                res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                score = 0
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break
            final_results['recommend'].append(score)
        else:
            review = tool_input[instructions]
            for r in review:
                target_review = task['target']['review']['text']
                agent_review = r
                similarity = compute_similarity(target_review, agent_review)
            final_results['review'].append(similarity)

    combined_data = [
        ['Search', len(final_results['search']), sum(tool_accuracy['search']) / len(tool_accuracy['search']), sum(final_results['search']) / len(final_results['search'])],
        ['Recommend', len(final_results['recommend']), sum(tool_accuracy['recommend']) / len(tool_accuracy['recommend']), sum(final_results['recommend']) / len(final_results['recommend'])],
        ['Review', len(final_results['review']), sum(tool_accuracy['review']) / len(tool_accuracy['review']), sum(final_results['review']) / len(final_results['review'])],
        ['Overall', len(final_results['search'] + final_results['recommend'] + final_results['review']),
        sum(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) / len(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']),
        sum(final_results['search'] + final_results['recommend'] + final_results['review']) / len(final_results['search'] + final_results['recommend'] + final_results['review'])]
    ]

    headers = ['Task Type', 'Total', 'Tool Accuracy Avg', 'Result Avg']
    print(tabulate(combined_data, headers=headers, tablefmt='grid'))

