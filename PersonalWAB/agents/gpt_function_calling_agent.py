
import json
from typing import Dict, List

from openai import OpenAI
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.agents.utils import (
    message_to_action,
    message_to_dict,
    pretty_print_conversation,
    pretty_history,
    encode_texts,
    HISTORY_PROMPT,
    MINI_HISTORY_PROMPT,
    INTEREC_MEMORY_PROMPT,
    RECMIND_ST_PROMPT,
    RECMIND_MT_PROPMT,
    INTEREC_PROMPT,
    INTEREC_UPDATE_MEM_PROMPT,
    TS_AGENT_PROMPT,
    TS_AGENT_MT_PROMPT,
    sup_search_pretty_history,
    sup_rec_pretty_history,
    sup_review_pretty_history,
    interecagent_pretty_history,
    mini_pretty_history,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random


client = None
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

def initialize_client(**kwargs):
    global client
    client = OpenAI(**kwargs)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def chat_completion_request(
    messages: List[Dict[str, str]],
    model: str,
    tools=None,
    tool_choice="auto",
    temperature: float = 0.0,
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
    )
    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        tool_call = message.tool_calls[0]
        json.loads(tool_call.function.arguments)
    return message, dict(response.usage)


class GPTFunctionCallingAgent(BaseAgent):
    def __init__(self, tools, sys_prompt, model: str = "gpt-4o-mini", function_selection_file=None, memory_file=None):
        self.tools = tools
        self.sys_prompt = sys_prompt
        self.model = model
        self.function_selection_file = json.load(open(function_selection_file)) if function_selection_file is not None else None
        self.memory_file = memory_file if memory_file is not None else None
        self.interecagent_memory = json.load(open(memory_file)) if memory_file is not None else None
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}
        self.reset()

    def reset(self, memory='none'):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        if memory != 'none':
            self.messages = [{"role": "system", "content": self.sys_prompt + memory}]
        self.usage = {"completion_tokens": [], "prompt_tokens": [], "total_tokens": []}

    def act(self, env, index=None, verbose=False, temperature=0.0, max_steps=30, memory='none', memory_length=10):

        if memory != 'none':
            memory_content = self.retrieve_memory(env, index, memory, memory_length)
        else:
            memory_content = 'none'
        
        if memory == 'recmind':
            if max_steps == -1:
                self.sys_prompt = RECMIND_ST_PROMPT
            else:
                self.sys_prompt = RECMIND_MT_PROPMT.replace("<NUM>", str(max_steps))
        if memory == 'interecagent':
            self.sys_prompt = INTEREC_PROMPT.replace("<NUM>", str(max_steps))
        
        if memory == 'taskspe':
            if max_steps == -1:
                self.sys_prompt = TS_AGENT_PROMPT
            else:
                self.sys_prompt = TS_AGENT_MT_PROMPT.replace("<NUM>", str(max_steps))

        self.reset(memory_content)
        obs, info = env.reset(index=index)

        max_steps = max_steps if max_steps > 0 else 10
        action_acc = []
        res_acc = []

        self.messages.append({"role": "user", "content": obs})

        if verbose:
            self.render(1)
        for _ in range(max_steps):
            message, usage = chat_completion_request(
                self.messages,
                model=self.model,
                tools=self.tools,
                temperature=temperature,
            )
            for key, value in usage.items():
                if key == 'completion_tokens_details' or key == 'prompt_tokens_details':
                    continue
                self.usage[key].append(value)
            if isinstance(message, Exception) and "context_length_exceeded" in str(
                message
            ):
                print(message)
                info["error"] = str(message)
                break
            action = message_to_action(message)

            obs, res, done, info = env.step(action)

            if action["name"] == "respond":
                self.messages.append({"role": "assistant", "content": message.content})
                self.messages.append({"role": "user", "content": obs})
            else:
                message.tool_calls = message.tool_calls[:1]
                self.messages.append(message)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_calls[0].id,
                        "name": message.tool_calls[0].function.name,
                        "content": obs,
                    }
                )
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
                
        if memory == 'interecagent':
            self.update_interecagent_memory(env, index, self.get_messages())
        
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

    def get_messages(self) -> List[Dict[str, str]]:
        return [message_to_dict(message) for message in self.messages]
    
    def retrieve_memory(self, env, index, memory, memory_length):
        if memory == 'last':
            '''Short Memory of This User (most recent purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True) 
            history = history[:memory_length]
            mem = '\nYour Memory of This User (most recent user past purchases and reviews, sorted by timestamp, most recent first):\n\n'
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
        
        elif memory == 'recmind':
            '''RecMind Memory (all purchases and reviews)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)[:memory_length]
            mem = '\nPersonalized Memory of This User (all user past purchases and reviews):\n\n'
            mem = mem + "\n".join([mini_pretty_history(item, i) for i, item in enumerate(history)])
            return mem

        elif memory == 'interecagent':
            '''InterecAgent Memory (history, like, dislike and expect)'''
            timestamp = env.tasks[index]["timestamp"]
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            history = sorted(history, key=lambda x: x['review']["timestamp"], reverse=True)[:memory_length]
            history = [interecagent_pretty_history(item) for item in history]
            assert self.interecagent_memory is not None
            mem = "\n\nYour Memory of This User:\nHistory:\n" 
            mem = mem + "\n".join(history) 
            mem = mem + "\nLike:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["like"])
            mem = mem + "\nDislike:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["dislike"])
            mem = mem + "\nExpect:\n" + str(self.interecagent_memory[env.tasks[index]["user_id"]]["expect"])
            return mem
    
        elif memory == 'taskspe':
            '''Task-specific Memory'''
            timestamp = env.tasks[index]["timestamp"]
            assert self.function_selection_file is not None
            tool_name = self.function_selection_file[env.tasks[index]["task"]][0] 
            if tool_name == 'search_product_by_query':
                task_type = 'search'
            elif tool_name == 'get_recommendations_by_history':
                task_type = 'recommend'
            elif tool_name == 'add_product_review':
                task_type = 'review'   
            
            history = env.init_data["user_history"][env.tasks[index]["user_id"]]
            history = [item for item in history if item['review']["timestamp"] < timestamp]
            if len(history) == 0:
                return 'none'
            history = self.retrieve_top_k_memories(env.tasks[index]['task'], history, sim_model, sim_tokenizer, k=memory_length)
            history = self.build_taskspe_memory(history, task_type)
            
            mem = f'- The task type is {task_type}\n- Use {tool_name} tool to complete task\n'
            mem = mem + '- Task-specific Memory:\n'
            mem = mem + "|".join([item for item in history])
            return mem

        else:
            return 'none'
    
    
    def update_interecagent_memory(self, env, index, messages):
        assert self.interecagent_memory is not None
        assert self.memory_file is not None
        user_id = env.tasks[index]["user_id"]
        prompt = INTEREC_UPDATE_MEM_PROMPT
        actions = ""
        for mes in messages[1:]:
            if mes["role"] == "user"or mes["role"] == "tool":
                actions += mes["role"] +':'+ mes["content"] + "\n"
            elif mes["role"] == "assistant":
                if 'function_call' in mes:
                    actions += 'assistant:' + mes["function_call"] + "\n"
                else:
                    actions += 'assistant:' + mes["content"] + "\n"
        prompt = prompt.replace("{conversation}", actions)

        client = OpenAI()

        profile = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        ).choices[0].message.content
        profile = profile.replace("```json", "").replace(
                "```", "").replace("\n", "")
        profile = json.loads(profile)

        cur_memory = self.interecagent_memory[user_id]
        like_set = set(cur_memory['like'])
        dislike_set = set(cur_memory['dislike'])
        expect_set = set(cur_memory['expect'])
        like_set -= set(profile.get('dislike', []))
        like_set.update(profile.get('like', []))
        dislike_set -= set(profile.get('like', []))
        dislike_set.update(profile.get('dislike', []))
        expect_set.update(profile.get('expect', []))
        if len(expect_set) > 30:
            expect_set = set(list(expect_set)[-30:])
        self.interecagent_memory[user_id] = {
            'like': list(like_set),
            'dislike': list(dislike_set),
            'expect': list(expect_set)
        }
        with open(self.memory_file, "w") as f:
            json.dump(self.interecagent_memory, f, indent=2)
        
        
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


    def build_taskspe_memory(self, history, task_type):
        if len(history) == 0:
            return ''
        if task_type == 'search':
            #history = sorted(history, key=lambda x: x['review']["timestamp"])
            history = [sup_search_pretty_history(item) for item in history]
            return history
        elif task_type == 'recommend':
            history = sorted(history, key=lambda x: x['review']["timestamp"])
            history = [sup_rec_pretty_history(item) for item in history]
            return history
        elif task_type == 'review':
            #history = sorted(history, key=lambda x: x['review']["timestamp"])
            history = [sup_review_pretty_history(item) for item in history]
            return history
        else:
            raise ValueError(f"Unknown task type: {task_type}")


sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to('cuda')


