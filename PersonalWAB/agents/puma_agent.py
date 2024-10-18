
import json
from typing import Dict, List

from openai import OpenAI
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.agents.utils import (
    message_to_action,
    message_to_dict,
    pretty_print_conversation,
    load_input_prompt,
    encode_texts,
    pretty_history,
    sup_search_pretty_history,
    sup_rec_pretty_history,
    sup_review_pretty_history,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random

from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from peft import PeftModel

generation_config = GenerationConfig(
                    num_beams=1,
                    max_new_tokens=256,
                    num_return_sequences=1,
                    #early_stopping=True,
                    use_cache=True,
                )

class PUMAAgent(BaseAgent):
    def __init__(self, function_selection_file, res_file, function_param_model, sys_prompt, tokenizer, max_length=1024, memory_token_length=512):
        self.sys_prompt = sys_prompt # we do not use this in PUMA agent
        self.function_selection_file = json.load(open(function_selection_file))
        self.res_file = json.load(open(res_file)) if res_file is not None else None
        self.function_param_model = function_param_model  
        self.tokenizer = tokenizer  
        self.max_length = max_length
        self.memory_token_length = memory_token_length
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}

    def act(self, env, index=None, verbose=False, temperature=0.0, max_steps=30, memory='none', memory_length=10):
        if memory != 'none':
            memory_content = self.retrieve_memory(env, index, memory, memory_length)

        self.reset()
        obs, info = env.reset(index=index)

        max_steps = max_steps if max_steps > 0 else 10
        action_acc = []
        res_acc = []

        self.messages.append({"role": "user", "content": obs})

        if verbose:
            self.render(1)
        for _ in range(max_steps):
 
            function = self.select_function(env, index)[0]
            if self.res_file is not None:
                function_input = self.res_file[env.tasks[index]['task']]
            else:
                obs = self.preprocess_observation(env, index, function, memory_content, self.memory_token_length)
                function_input = self.generate_function_param_with_llama(obs)
            
            if function_input.startswith('Say:'):
                action = {"name": "respond", "arguments": {"content": function_input.split('Say: ')[-1]}}
            elif function_input.startswith('stop'):
                function = 'stop'
                action = {"name": "stop", "arguments": {}}
            else:
                if function == "get_recommendations_by_history":
                    function_input = [item.strip() for item in function_input.split(',')]
                    function_input = list(set(function_input))
                    action = {"name": function, "arguments": {"product_sequence": function_input}}
                elif function == "search_product_by_query":
                    action = {"name": function, "arguments": {"query": function_input}}
                elif function == "add_product_review":
                    action = {"name": function, "arguments": {"review": function_input}}
                
            for key, value in self.usage.items():
                if key == 'completion_tokens_details':
                    continue
                self.usage[key].append(0)

            obs, res, done, info = env.step(action)

            if action["name"] == "respond":
                self.messages.append({"role": "assistant", "content": function_input})
                self.messages.append({"role": "user", "content": obs})
            else:
                self.messages.append({"role": "assistant", "content": function_input})
                self.messages.append({"role": "tool", "name": function, "content": obs})

            if verbose:
                self.render(2)

            if env.max_steps == -1 and done:
                action_acc.append(res[0])
                res_acc.append(res[1])
                break
            else:
                if done:
                    action_acc.append(res[0])
                    res_acc.append(res[1])
                if action["name"] == "stop":
                    break

        info["usage"] = self.usage
        return action_acc, res_acc, info

    def select_function(self, env, index):
        function = self.function_selection_file[env.tasks[index]['task']]
        # may generate by trained llama as well, but we pre-generate to save time
        return function

    def generate_function_param_with_llama(self, observation):
        """Use the LLaMA model to generate the function param."""
        input_text = observation
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = self.function_param_model.generate(inputs['input_ids'], generation_config=generation_config)
        param = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split('### Tool Parameter:\n')[-1]
        return param

    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        return [message_to_dict(message) for message in self.messages]
    
    def retrieve_memory(self, env, index, memory, memory_length):
        if memory == 'taskspe':
            '''Task-spcific Memory (all purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            task_type = env.tasks[index]["type"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            if len(history) == 0:
                return 'none'
            history = self.retrieve_top_k_memories(env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=memory_length)
            history = self.build_taskspe_memory(history, task_type)
            mem = "|".join([item for item in history])
            return mem
        
        else:
            return 'none'

    def build_taskspe_memory(self, history, task_type):
        if len(history) == 0:
            return ''
        if task_type == 'search':
            history = [sup_search_pretty_history(item) for item in history]
            return history
        elif task_type == 'recommend':
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True) # to keep the latest items in trucated memory
            history = [sup_rec_pretty_history(item) for item in history]
            return history
        elif task_type == 'review':
            history = [sup_review_pretty_history(item) for item in history]
            return history
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def retrieve_top_k_memories(self, request, history, model, tokenizer, k=50):
        '''Retrieve top k relevant memories'''
        request_embedding = encode_texts([request], model, tokenizer)
        history_embeddings = encode_texts([pretty_history(item, i) for i, item in enumerate(history)], model, tokenizer)
        similarity = F.cosine_similarity(request_embedding, history_embeddings, dim=1)
        top_k = similarity.argsort(descending=True)[:k]
        del request_embedding, history_embeddings, similarity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [history[i] for i in top_k]
    
    def preprocess_observation(self, env, index, tool, memory_content, memory_length):
        if tool == "search_product_by_query":
            task_type = "search"
        elif tool == "get_recommendations_by_history":
            task_type = "recommend"
        else:
            task_type = "review"
        input_prompt = load_input_prompt(env.tasks[index]['task'], task_type, env.tasks[index]['target']['product_info'],
                                            memory_content, self.tokenizer, memory_length) 

        return input_prompt


sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')
