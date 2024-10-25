import os
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    GenerationConfig,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from typing import Dict, List
import torch.nn.functional as F
import sys
import json
import re
from datasets import load_dataset


class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, json_file, max_length, split="train", subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset_size = subset_size
        
        with open(json_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.dataset = data.get(split, [])
        if self.subset_size is not None:
            indices = list(range(len(self.dataset)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.dataset = [self.dataset[i] for i in sampled_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_text = item['prompt']
        target_text = item['target']
        return preprocess_function(input_text, target_text, self.tokenizer, self.max_length)


def preprocess_function(prefix_text, target_text, tokenizer, max_length):
    
    response_text = f"{prefix_text}{target_text}</s>"

    input = tokenizer(
        response_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        padding="max_length",  
    )
    
    input_ids = input["input_ids"]
    labels = input_ids.copy()

    prefix_tex = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    output_start_index = len(prefix_tex)

    labels[:output_start_index] = [-100] * output_start_index  

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }


def load_function_prompt(data_file, split):
    data = json.load(open(data_file, encoding="utf-8"))
    tasks = []
    source = []
    target = []
    for i in range(len(data[split])):
        task= data[split][i]['instruction']
        source_text = data[split][i]['prompt']
        target_text = data[split][i]['target']
        source.append(source_text)
        target.append(target_text)
        tasks.append(task)
    return tasks, source, target


def load_param_prompt(data_file, tool_file, split, mem_token_length, tokenizer):
    data = json.load(open(data_file, encoding="utf-8"))
    tool_file = json.load(open(tool_file))
    tasks = []
    source = []
    target = []
    for i in range(len(data[split])):
        item = data[split][i]
        task = item['instruction']
        input_text = PARAM_PROMPT.replace('<Instruction>', task)
        mem = item['mem']
        tokenized_memory = tokenizer(mem, return_tensors=None, truncation=True, max_length=mem_token_length)
        memory_text = tokenizer.decode(tokenized_memory["input_ids"], skip_special_tokens=True)
        input_text = input_text.replace('<Memory>', memory_text)
        input_text = input_text.replace('<Tool>', tool_file[task][0])
        tool_input = item['target']
        
        tasks.append(task)
        source.append(input_text)
        target.append(tool_input)
    
    return tasks, source, target


class LlaMaTrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, vocab_size=32000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits / self.temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        #print(shift_logits.shape, shift_labels.shape)
        loss = loss_fct(shift_logits, shift_labels)
        torch.set_printoptions(threshold=torch.inf)
        # if loss == float('inf') or torch.isnan(loss):
        #     print('inf or nan loss')
        #     print(inputs.input_ids)

        return (loss, outputs) if return_outputs else loss


from openai import OpenAI

HISTORY_PROMPT = '''
MEMORY <NUM>:
Product:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Details: <DETAILS>
- Description: <DESCRIPTION>
- Features: <FEATURES>
Review:
- Rating: <RATING>
- Text: <TEXT>
- Timestamp: <TIMESTAMP>
'''

TS_SEARCH_PROMPT = '''Title:<TITLE>
Main Category:<MAIN_CATEGORY>
Price:<PRICE>
Store:<STORE>
'''

TS_REC_PROMPT = '''Title:<TITLE>
Main Category:<MAIN_CATEGORY>
Asin:<ASIN>
'''

TS_REV_PROMPT = '''Rating:<RATING>
Text:<TEXT>
'''

PRODUCT_INFO_PROMPT = '''
The product is:
Title: <Title>
Price: <Price>
Store: <Store>
Main Category: <Main Category>
'''

PARAM_PROMPT = '''Below is an instruction that describes a task. Generate the tool parameter that appropriately completes the request. 
### Instruction:<Instruction>
  
Memory: <Memory>

Tool: <Tool>

### Tool Parameter:
'''

FUNCTION_PROMPT = '''Below is an instruction that describes a task. Choose a tool that appropriately completes the request.
### Instruction: <Instruction>

### Tool:
'''

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        if torch.cuda.is_available():
            encoded_input = encoded_input.to('cuda')

        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        all_embeddings.append(sentence_embeddings)

        torch.cuda.empty_cache()  # Clear memory cache
    return torch.cat(all_embeddings, dim=0)

def pretty_history(item, num):
    res = HISTORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<AVERAGE_RATING>", str(item['product_info']['average_rating']))
    res = res.replace("<RATING_NUMBER>", str(item['product_info']['rating_number']))
    res = res.replace("<PRICE>", str(item['product_info']['price']))
    res = res.replace("<STORE>", str(item['product_info']['store']))
    res = res.replace("<DETAILS>", json.dumps(item['product_info']['details']))
    res = res.replace("<DESCRIPTION>", str(item['product_info']['description']))
    res = res.replace("<FEATURES>", str(item['product_info']['features']))
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    res = res.replace("<TIMESTAMP>", str(item['review']['timestamp']))
    res = res.replace("<NUM>", str(num))
    return res

def retrieve_top_k_memories(request, history, model, tokenizer, k=50):
    request_embedding = encode_texts([request], model, tokenizer)
    history_embeddings = encode_texts([pretty_history(item, i) for i, item in enumerate(history)], model, tokenizer)
    similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
    top_k = similarity.argsort(descending=True)[:k]
    torch.cuda.empty_cache()
    return [history[i] for i in top_k]

def prettify_product_info(product_info):
    res = PRODUCT_INFO_PROMPT.replace('<Title>', product_info['title'])
    res = res.replace('<Price>', str(product_info['price']))
    res = res.replace('<Store>', str(product_info['store']))
    res = res.replace('<Main Category>', str(product_info['main_category']))
    return res

def sup_pretty_history(item, task_type):
    if task_type == 'search':
        return TS_SEARCH_PROMPT.replace("<TITLE>", item['product_info']['title']).replace("<MAIN_CATEGORY>", str(item['product_info']['main_category'])).replace("<PRICE>", str(item['product_info']['price'])).replace("<STORE>", str(item['product_info']['store']))
    elif task_type == 'recommend':
        return TS_REC_PROMPT.replace("<TITLE>", item['product_info']['title']).replace("<MAIN_CATEGORY>", str(item['product_info']['main_category'])).replace("<ASIN>", str(item['product_info']['parent_asin']))
    elif task_type == 'review':
        return TS_REV_PROMPT.replace("<RATING>", str(item['review']['rating'])).replace("<TEXT>", item['review']['text'])

def build_taskspe_memory(history, task_type):
    if len(history) == 0:
        return []
    if task_type == 'recommend':
        history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)
    return [sup_pretty_history(item, task_type) for item in history]

def generate_search_query(instruction, mem):
    prompt = '''As a personalized shopping agent, you can help users search for products.

Rules:
- The user will provide a request.
- You need to use the tool to find the product, the params for the tool is a textual query.
- Make the best tool call based on the user's request and the memory provided.
- Information in the memory can help you make a better tool call.
- You have only one chance to make a tool call, so make sure you have the best input for the tool.
- The tool will be provided, you only need to provide the most appropriate input for the tool.
- Do not inlcude any tool name, other information, or explanation.

Instruction: <Instruction>

Memory: <Memory>

Tool: search_product_by_query
'''
    prompt = prompt.replace('<Instruction>', instruction)
    prompt = prompt.replace('<Memory>', '|'.join(mem))   
    messages = [{'role': 'system', 'content': prompt}]    
    #print(prompt)
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0,
    )
    message = response.choices[0].message.content
    #print(message)
    return message

    