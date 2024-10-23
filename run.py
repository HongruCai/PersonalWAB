
from scipy.special import comb
from typing import Any, Dict
import os
import json
import random
import argparse
import multiprocessing
import sys
from math import comb
from typing import Any, Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import torch
import time
from PersonalWAB.agents.base import BaseAgent
from PersonalWAB.envs import get_env


def run(
    args: argparse.Namespace,
    ckpt_path,
):
    if args.max_steps == -1 and args.user_mode != "no":
        raise ValueError("Max steps must be set for user simulation mode")

    env = get_env(
        args.env,
        user_mode=args.user_mode,
        user_model=args.user_model,
        task_split=args.task_split,
        max_steps=args.max_steps,
    )
    end_index = (
        len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    )
    results = []
    lock = multiprocessing.Lock()
    print(
        f"Running {args.task_split} tasks {args.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(args.num_trials):
        idxs = list(range(args.start_index, end_index))
        finished_idxs = []
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                finished_tasks = json.load(f)
                for res in finished_tasks:
                    if 'task_id' in res:
                        results.append(res)
                        finished_idxs.append(res["task_id"])
        idxs = [idx for idx in idxs if idx not in finished_idxs]


        def _run(idx: int) -> dict:
            isolated_env = get_env(
                args.env,
                user_mode=args.user_mode,
                user_model=args.user_model,
                task_split=args.task_split,
                max_steps=args.max_steps,
            )

            isolated_agent = agent_factory(
                tools_info=env.functions_info,
                sys_prompt=env.sys_prompt,
                args=args,
            )

            action_acc, res_acc, info = isolated_agent.act(
                isolated_env,
                idx,
                verbose=args.verbose,
                temperature=args.temperature,
                max_steps=env.max_steps,
                memory=args.agent_memory,
            )
            
            result = {
                "task_id": idx,
                "action_acc": action_acc,
                "res_acc": res_acc,
                "info": info,
                "traj": isolated_agent.get_messages(),
                "trial": i,
            }

            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                else:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            for res in tqdm(executor.map(_run, idxs), total=len(idxs), desc=f"Trial {i}"):
                results.append(res)
    return results


def agent_factory(tools_info, sys_prompt, args: argparse.Namespace) -> BaseAgent:

    if args.agent_strategy == "function_calling":
        tools_info = [
            tool for tool in tools_info if tool["function"]["name"] != "get_product_details_by_asin"
        ]
        if (
            "gpt" in args.model
            or "mistralai/Mi" in args.model
            or "meta-llama/Meta-Llama-3-" in args.model
        ):
            from PersonalWAB.agents.gpt_function_calling_agent import (
                GPTFunctionCallingAgent,
                initialize_client,
            )

            if "gpt" in args.model:
                initialize_client(
                    api_key=os.getenv("OPENAI_API_KEY")
                )

            return GPTFunctionCallingAgent(tools_info, sys_prompt, model=args.model, 
                                           function_selection_file=args.puma_function_file, memory_file=args.interec_memory_file)

    elif args.agent_strategy == "react" or args.agent_strategy == "react_reflect":
        tools_info = [
            tool for tool in tools_info if tool["function"]["name"] != "get_product_details_by_asin"
        ]
        from PersonalWAB.agents.chat_react_agent import ChatReActAgent, initialize_create

        if "gpt" in args.model:
            initialize_create(mode="openai")
        if args.agent_strategy == "react":
            return ChatReActAgent(tools_info, sys_prompt, model=args.model, )
        elif args.agent_strategy == "react_reflect":
            return ChatReActAgent(tools_info, sys_prompt, model=args.model, reflection=True)
    elif args.agent_strategy == "recmind":

        if (
            "gpt" in args.model
            or "mistralai/Mi" in args.model
            or "meta-llama/Meta-Llama-3-" in args.model
        ):
            from PersonalWAB.agents.gpt_function_calling_agent import (
                GPTFunctionCallingAgent,
                initialize_client,
            )

            if "gpt" in args.model:
                initialize_client(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif (
                "mistralai/Mi" in args.model or "meta-llama/Meta-Llama-3-" in args.model
            ):
                initialize_client(
                    api_key=os.getenv("ANYSCALE_API_KEY"),
                    base_url="https://api.endpoints.anyscale.com/v1",
                )

            return GPTFunctionCallingAgent(tools_info, sys_prompt, model=args.model)
    
    elif args.agent_strategy == "puma":
        from PersonalWAB.agents.puma_agent import PUMAAgent
        function_file = args.puma_function_file
        param_file = args.puma_param_file
        if args.puma_generate == 0:
            '''To save time, simply use pre-generated results to evaluate'''
            return PUMAAgent(function_file, param_file, None, sys_prompt, None)
        #If you want to generate results, you need to change the path to the model
        llama_model, llama_tokenizer = load_llama_model(args.puma_model_path, 'meta-llama/Llama-2-7b-chat-hf', torch.float16)
        return PUMAAgent(function_file, None, llama_model, sys_prompt, llama_tokenizer, max_length=1024, memory_token_length=args.mem_token_length)
    else:
        raise ValueError(f"Unknown agent strategy: {args.agent_strategy}")


global_model = None
global_tokenizer = None
def load_llama_model(model_path, base_model, torch_dtype):
    global global_model, global_tokenizer
    if global_model is None and global_tokenizer is None:
        global_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        global_model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                #device_map=device_map,
            )
        global_model = PeftModel.from_pretrained(
                global_model,
                model_path,
                torch_dtype=torch_dtype,
                #device_map=device_map,
            )

        global_tokenizer.padding_side = "left"
        if torch.cuda.is_available():
            global_model.to('cuda')
    return global_model, global_tokenizer


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_types = ["search", "recommend", "review"]
    
    stats = {
        task_type: {
            "action_sum": 0,
            "res_sum": 0,
            "total_count": 0,
            "interaction_count": 0  
        }
        for task_type in task_types
    }
    
    global_stats = {
        "action_sum": 0,
        "res_sum": 0,
        "total_count": 0,
        "interaction_count": 0  
    }

    for result in results:

        if 'info' not in result:
            continue
        task_type = result.get("info", {}).get("task", {}).get("type")
        
        interaction_count = len(result['info']['usage']['completion_tokens'])
        stats[task_type]["interaction_count"] += interaction_count
        global_stats["interaction_count"] += interaction_count

        action_acc_list = result.get("action_acc", [0])
        res_acc_list = result.get("res_acc", [0])

        if len(action_acc_list) == 0 or len(res_acc_list) == 0:
            action_acc_list = [0]
            res_acc_list = [0]
        
        valid_indexes = [i for i, acc in enumerate(action_acc_list) if acc == 1]

        if valid_indexes:
            best_index = max(valid_indexes, key=lambda i: res_acc_list[i])
            best_action_acc = action_acc_list[best_index]
            best_res_acc = res_acc_list[best_index]

            stats[task_type]["action_sum"] += best_action_acc
            global_stats["action_sum"] += best_action_acc

            stats[task_type]["res_sum"] += best_res_acc
            global_stats["res_sum"] += best_res_acc
        else:
            stats[task_type]["action_sum"] += 0
            global_stats["action_sum"] += 0
            stats[task_type]["res_sum"] += 0
            global_stats["res_sum"] += 0

        stats[task_type]["total_count"] += 1
        global_stats["total_count"] += 1

    final_stats = {}
    for task_type in task_types:
        total_count = stats[task_type]["total_count"]
        interaction_count = stats[task_type]["interaction_count"]
        if total_count > 0 and interaction_count > 0:
            avg_action_acc = stats[task_type]["action_sum"] / total_count
            avg_res_acc = stats[task_type]["res_sum"] / total_count
            avg_interaction_times = interaction_count / total_count  
        else:
            avg_action_acc, avg_res_acc, avg_interaction_times = 0, 0, 0
        
        final_stats[task_type] = {
            "total_count": total_count,
            "avg_interaction_times": avg_interaction_times,
            "avg_action_acc": avg_action_acc,  
            "avg_res_acc": avg_res_acc  
        }
    
    global_total_count = global_stats["total_count"]
    global_interaction_count = global_stats["interaction_count"]
    if global_total_count > 0 and global_interaction_count > 0:
        global_avg_action_acc = global_stats["action_sum"] / global_total_count
        global_avg_res_acc = global_stats["res_sum"] / global_total_count
        global_avg_interaction_times = global_interaction_count / global_total_count  
    else:
        global_avg_action_acc, global_avg_res_acc, global_avg_interaction_times = 0, 0, 0
    
    final_stats["overall"] = {
        "total_count": global_total_count,
        "avg_interaction_times": global_avg_interaction_times, 
        "avg_action_acc": global_avg_action_acc,
        "avg_res_acc": global_avg_res_acc
    }
    
    return final_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["pwab"], default="pwab"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=[
            # openai api models
            "gpt-4-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-4o",
            "gpt-4o-mini",
            # custom models
            "finetune/llama",
        ],
    )
    parser.add_argument(
        "--user_mode",
        type=str,
        default="no",
        choices=["no", "naive", "human"],
    )
    parser.add_argument(
        "--user_model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        default="function_calling",
        choices=["function_calling", "react", "react_reflect", 'recmind', 'puma'],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--task_split", type=str, default="test", choices=["train", "test"]
    )
    parser.add_argument(
        "--agent_memory", type=str, default="none", choices=["taskspe", "last", "relevant", "random", "recmind", "interecagent", "none"]
    )
    parser.add_argument(
        "--memory_length", type=int, default=1, help="Max memory length"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max step number for agents to run, -1 for single round and no user simulation"
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--interec_memory_file", type=str, default='data/interecagent_memory_train.json')
    
    parser.add_argument("--puma_param_file", type=str, default=None)
    parser.add_argument("--puma_function_file", type=str, default='PUMA/output/res/tool_results.json')
    parser.add_argument("--puma_generate", type=int, default=0)
    parser.add_argument("--puma_model_path", type=str, default='finetune/output/input/Llama-2-7b-chat-hf/')
    parser.add_argument("--mem_token_length", type=int, default=768)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    time_str = datetime.now().strftime("%m%d%H%M")
    turn_sig = 'singleturn' if args.max_steps == -1 else 'multiturn'
    file_str = f'''{args.log_dir}/{turn_sig}/step{args.max_steps}_{args.agent_strategy}-{args.model.split('/')[-1]}-{args.temperature}_mem{args.agent_memory}_range{args.start_index}-{args.end_index}_user{args.user_model}_{time_str}.json'''
    if args.resume_from:
        file_str = args.resume_from
        print(f"Resuming from {file_str}")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    results = run(
        args=args,
        ckpt_path=file_str,
    )

    final_res = calculate_statistics(results)
    for task_type, stats in final_res.items():
        print(f"\nTask type: {task_type}")
        for key, value in stats.items():
            print(f"{key}: {value}")

    total = {'run_args': vars(args), 'total cost': 0, 'results': final_res}
    for r in results:
        if 'total_price' in r['info']['usage']:
            total['total cost'] += r['info']['usage']['total_price']
        else:
            total['total cost'] += 0
    results.insert(0, total)

    with open(file_str, "w") as f:
        json.dump(results, f, indent=2)
        print(f"\n Results saved to {file_str}\n")
        print(f"Total cost: {total['total cost']}")


if __name__ == "__main__":
    main()
