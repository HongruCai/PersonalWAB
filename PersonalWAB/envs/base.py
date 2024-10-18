import sys
import random
from copy import deepcopy
from hashlib import sha256
from typing import Any, Callable, Dict, List, Tuple, TypedDict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from PersonalWAB.envs.user import load_user

sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
if torch.cuda.is_available():
    sim_model.to('cuda')


def compute_similarity(target_review, agent_review):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]

    encoded_input = sim_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    if torch.cuda.is_available():
        encoded_input.to('cuda')

    with torch.no_grad():
        model_output = sim_model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    del model_output
    del sentence_embeddings
    torch.cuda.empty_cache()

    return similarity
    

class Action(TypedDict):
    name: str
    arguments: Dict[str, Any]


def to_hashable(item):
    """
    Recursively converts an item into a hashable type. This function handles
    dictionaries, lists, and basic nested structures.
    """
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    else:
        return item


def consistent_hash(value):
    return sha256(str(value).encode("utf-8")).hexdigest()


class PWABaseEnv:
    def __init__(
        self,
        data: Dict[str, Any],
        functions: List[Callable],
        tasks: List[Dict[str, Any]],
        user_mode: str,
        user_model: str,
        sys_prompt: str,
        max_steps: int,
    ):
        super().__init__()
        self.init_data = data
        self.data = None
        self.functions = functions
        self.functions_dict = {tool.__name__: tool for tool in functions}
        self.functions_info = [tool.__info__ for tool in functions]
        self.action_functions = ['search_product_by_query',
                             'get_recommendations_by_history',
                             'add_product_review']
        self.tasks = tasks
        self.task = {}
        self.user = load_user(user_mode, user_model)
        self.actions = []
        self.index = None
        self.sys_prompt = sys_prompt
        self.max_steps = max_steps

    def reset(self, index=None, obs=True) -> Tuple[str, Dict[str, Any]]:
        if index is None:
            index = random.randint(0, len(self.tasks))
        self.index = index
        self.data = deepcopy(self.init_data)
        self.task = self.tasks[index]
        self.actions = []
        observation = (
            self.user.reset(instruction=self.task, data=self.data, task_type=self.task['type']) if obs else ""
        )
        return observation, {"task": self.task, "source": "user"}

    def step(self, action: Action) -> Tuple[str, list, bool, Dict[str, Any]]:
        if not isinstance(action, dict):
            raise TypeError("action must be a dictionary")
        if "name" not in action or not isinstance(action["name"], str):
            raise ValueError("action: 'name' key must be present and must be a string")
        if "arguments" not in action or not isinstance(action["arguments"], dict):
            raise ValueError(
                "action: 'arguments' key must be present and must be a dictionary"
            )

        self.actions.append(action)

        if action["name"] == "respond":
            observation = self.user.step(action["arguments"]["content"])
            res, done, info = [0,0], False, {"source": "user"}
            if observation == "Error: ###STOP###":
                done = True
        elif action["name"] in self.functions_dict:
            try:
                observation = self.functions_dict[action["name"]](
                    data=self.data, **action["arguments"]
                )
            except Exception as e:
                observation = f"Error: {e}"

            res, done, info = [0,0], False, {"source": action["name"]}
            if action["name"] in self.action_functions:
                done = True
        else:
            observation = f"Unknown action {action['name']}"
            res, done, info = [0,0], False, {"source": action["name"]}

        if done:
            info = {
            "task": {'user_id': self.task['user_id'], 
                     'task': self.task['task'], 
                     'type': self.task['type'],
                     'timestamp': self.task['timestamp'],
                     'target': self.task['target']['product_info']['parent_asin']},
            "actions": [
                action for action in self.actions if action["name"] != "respond"
            ],}
            if isinstance(observation, str) and observation.startswith("Error:"):
                return str(observation), res, done, info

            res = self.calculate_reward(action['name'], observation)
        info = {
            "task": {'user_id': self.task['user_id'], 
                     'task': self.task['task'], 
                     'type': self.task['type'],
                     'timestamp': self.task['timestamp'],
                     'target': self.task['target']['product_info']['parent_asin']},
            "actions": [
                action for action in self.actions if action["name"] != "respond"
            ],}
        info["user_cost"] = self.user.get_total_cost()    
        return str(observation), res, done, info

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    def calculate_reward(self, action, observation):
        res = [0, 0.0]
        if self.task['type'] == 'search':
            if action == 'search_product_by_query':
                res[0] = 1
            target_asin = self.task['target']['product_info']['parent_asin']
            if isinstance(observation, list):
                for i in range(len(observation)):
                    if target_asin in observation[i]:
                        res[1] = 1 - i/len(observation)
                        break

        elif self.task['type'] == 'recommend':
            if action == 'get_recommendations_by_history':
                res[0] = 1
            target_asin = self.task['target']['product_info']['parent_asin']
            if isinstance(observation, list):
                for i in range(len(observation)):
                    if target_asin in observation[i]:
                        res[1] = 1 - i/len(observation)
                        break

        elif self.task['type'] == 'review':
            if action == 'add_product_review':
                res[0] = 1
            if isinstance(observation, Dict):
                target_review = self.task['target']['review']['text']
                agent_review = observation['review']
                similarity = compute_similarity(target_review, agent_review)
                res[1] = similarity
        
        return res


class BaseEnv:
    def __init__(
        self,
        data: Dict[str, Any],
        tools: List[Callable],
        tasks: List[Dict[str, Any]],
        wiki: str,
        rules: List[str],
        user_mode: str,
        user_model: str,
    ):
        super().__init__()
        self.init_data = data
        self.data = None
        self.tools = tools
        self.tools_dict = {tool.__name__: tool for tool in tools}
        self.tools_info = [tool.__info__ for tool in tools]
        self.terminate_tools = []
        self.tasks = tasks
        self.task = None
        self.wiki = wiki
        self.rules = rules
        self.user = load_user(user_mode, user_model)
        self.actions = []
        self.index = None

    def reset(self, index=None, obs=True) -> Tuple[str, Dict[str, Any]]:
        if index is None:
            index = random.randint(0, len(self.tasks))
        self.index = index
        self.data = deepcopy(self.init_data)
        self.task = self.tasks[index]
        self.actions = []  # store the actions from the agent
        observation = (
            self.user.reset(instruction=self.task["instruction"]) if obs else ""
        )
        return observation, {"task": self.task, "source": "user"}

    def step(self, action: Action) -> Tuple[str, float, bool, Dict[str, Any]]:
        if not isinstance(action, dict):
            raise TypeError("action must be a dictionary")
        if "name" not in action or not isinstance(action["name"], str):
            raise ValueError("action: 'name' key must be present and must be a string")
        if "arguments" not in action or not isinstance(action["arguments"], dict):
            raise ValueError(
                "action: 'arguments' key must be present and must be a dictionary"
            )

        self.actions.append(action)

        if action["name"] == "respond":
            observation = self.user.step(action["arguments"]["content"])
            reward, done, info = 0, False, {"source": "user"}
            if observation == "###STOP###":
                done = True
        elif action["name"] in self.tools_dict:
            try:
                observation = self.tools_dict[action["name"]](
                    data=self.data, **action["arguments"]
                )
            except Exception as e:
                observation = f"Error: {e}"
            reward, done, info = 0, False, {"source": action["name"]}
            if action["name"] in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action['name']}"
            reward, done, info = 0, False, {"source": action["name"]}

        if done:
            reward, info = self.calculate_reward()
            info["user_cost"] = self.user.get_total_cost()
        return str(observation), reward, done, info

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    def calculate_reward(self) -> Tuple[float, Dict[str, Any]]:
        data_hash = self.get_data_hash()
        reward, info = 1, {
            "data_hash": data_hash,
            "actions": [
                action for action in self.actions if action["name"] != "respond"
            ],
        }

        return reward, info
