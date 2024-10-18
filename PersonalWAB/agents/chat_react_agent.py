import json
from typing import Dict, List

from openai import OpenAI
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.agents.utils import (
    message_to_action,
    message_to_dict,
    pretty_print_conversation,
    pretty_history,
    REACT_INST,
    REFLECTION_INST,
    encode_texts,
    HISTORY_PROMPT,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time
import random


create = None
create_mode = None
prompt_price_per_million = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 5,
    "gpt-4-turbo": 10,
    "gpt-4-32k-0613": 60,
    "gpt-3.5-turbo": 0.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}
completion_price_per_million = {
    "gpt-4o-mini": 0.60,
    "gpt-4o": 15,
    "gpt-4-turbo": 30,
    "gpt-4-32k-0613": 120,
    "gpt-3.5-turbo": 1.5,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.15,
    "meta-llama/Meta-Llama-3-70B-Instruct": 1.0,
}


def initialize_create(mode="openai", **kwargs):
    global create, create_mode
    if mode == "openai":
        from openai import OpenAI

        create = OpenAI(**kwargs).chat.completions.create
        create_mode = "openai"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def get_message_action(
    messages, model, **kwargs
):  # kwargs only contain temperature for now
    global create, create_mode
    if create_mode == "openai":
        kwargs["model"] = model
        kwargs["messages"] = messages

    response = create(**kwargs)

    if create_mode == "openai":
        message = response.choices[0].message.content

    action_name = message.split("Action:")[-1].split("Arguments:")[0].strip()
    action_args = message.split("Arguments:")[-1].strip().split("\n")[0]
    if action_name == "respond" or action_name == "":
        action_args = {"content": action_args}
    else:
        action_args = json.loads(action_args)
    return message, {"name": action_name, "arguments": action_args}, dict(response.usage)


class ChatReActAgent(BaseAgent):
    def __init__(self, tools, sys_prompt, model: str = "gpt-4o-mini", reflection: bool = False):
        instruction = REACT_INST 
        self.sys_prompt = sys_prompt + "\n#Available tools\n" + json.dumps(tools) + instruction 
        self.model = model
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.reset()
        self.reflection = reflection

    def reset(self, memory='none'):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        if memory != 'none':
            self.messages = [{"role": "system", "content": self.sys_prompt + memory}]
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}

    def act(self, env, index=None, verbose=False, temperature=0.0, max_steps=30, memory='none', memory_length=10):

        if memory != 'none':
            memory_content = self.retrieve_memory(env, index, memory, memory_length=memory_length)
            
        self.reset(memory_content)
        obs, info = env.reset(index=index)
  
        max_steps = max_steps if max_steps > 0 else 3
        action_acc = []
        res_acc = []

        self.messages.append({"role": "user", "content": obs})

        if verbose:
            self.render(1)
        for _ in range(max_steps):
            try:
                message, action, usage = get_message_action(
                    self.messages, self.model, temperature=temperature
                )
                for key, value in usage.items():
                    if key == 'completion_tokens_details' or key == 'prompt_tokens_details':
                        continue
                    self.usage[key].append(value)
            except Exception as e:
                print(e)
                info["error"] = str(e)
                break
            obs, res, done, info = env.step(action)
            if action["name"] != "respond":
                obs = "API output: " + obs
                #print(action["name"])
            self.messages.append({"role": "assistant", "content": message})
            self.messages.append({"role": "user", "content": obs})
            if self.reflection:
                if action["name"] == "respond":
                    reflection = self.generate_reflection(env.tasks[index]['task'], self.messages)
                    self.messages.append({"role": "assistant", "content": reflection})

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

        self.usage.update(
            {"completion_price": [], "prompt_price": [], "total_price": []}
        )
        self.usage["completion_price"] = (
            completion_price_per_million[self.model]
            * sum(self.usage["completion_tokens"])
            / 1e6
        )
        self.usage["prompt_price"] = (
            prompt_price_per_million[self.model]
            * sum(self.usage["prompt_tokens"])
            / 1e6
        )
        self.usage["total_price"] = (
            self.usage["completion_price"] + self.usage["prompt_price"]
        )
        info["usage"] = self.usage
        
        return action_acc, res_acc, info

    def render(self, last_n=None):
        if last_n is not None:
            pretty_print_conversation(self.messages[-last_n:])
        else:
            pretty_print_conversation(self.messages)

    def get_messages(self):
        return self.messages
    
    def retrieve_memory(self, env, index, memory, memory_length):
        if memory == 'last':
            '''Last Memory of This User (most recent purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True) 
            history = history[:memory_length]
            mem = '\n\nYour Memory of This User (most recent user past purchases and reviews, sorted by timestamp, most recent first):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
        
        elif memory == 'relevant':
            '''Relevant Memory of This User (most relevant purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = self.retrieve_top_k_memories(env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=memory_length)
            mem = '\nYour Memory of This User (most relevant user past purchases and reviews, sorted by relevance):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
        
        elif memory == 'random':
            '''Random Memory of This User (random purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)
            history = random.sample(history, min(memory_length, len(history)))
            mem = '\nYour Memory of This User (random chosed user past purchases and reviews):\n\n'
            mem = mem + "\n".join([pretty_history(item, i) for i, item in enumerate(history)])
            return mem
    
        else:
            return ''
    
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


    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
    def generate_reflection(self, task, messages):

        prompt = REFLECTION_INST.replace("<TASK>", task)
        actions = ""
        for mes in messages[1:-1]:
            actions += mes["role"] +':'+ mes["content"] + "\n"
        prompt = prompt.replace("{conversation}", actions)
        prompt = prompt.replace("<ACTIONS>", actions)
        prompt = prompt.replace("<FEEDBACK>", messages[-1]["content"])
        
        client = OpenAI()

        reflection = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        ).choices[0].message.content
        
        return reflection



sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')




