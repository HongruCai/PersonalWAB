
from math import comb
from typing import Any, Dict, List
import json


from typing import Any, Dict, List

from typing import Any, Dict, List

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_types = ["search", "recommend", "review"]
    
    # 初始化任务类型的统计数据
    stats = {
        task_type: {
            "action_sum": 0,
            "res_sum": 0,
            "total_count": 0,
            "interaction_count": 0  # 用于记录总交互次数
        }
        for task_type in task_types
    }
    
    global_stats = {
        "action_sum": 0,
        "res_sum": 0,
        "total_count": 0,
        "interaction_count": 0  # 全局交互次数统计
    }
    index_rec = []
    for result in results:
        #print(result['task_id'])
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
        
        # action = action_acc_list[0]
        # res = res_acc_list[0] if action == 1 else 0
        # stats[task_type]["action_sum"] += action
        # global_stats["action_sum"] += action

        # # 统计与最优 action_acc 对应的 res_acc
        # stats[task_type]["res_sum"] += res
        # global_stats["res_sum"] += res

        #找出 action_acc == 1 的位置
        valid_indexes = [i for i, acc in enumerate(action_acc_list) if acc == 1]

        # 如果没有 action_acc == 1 的情况，跳过 res_acc 的统计
        if valid_indexes:
            # 找到对应 action_acc == 1 时，res_acc 的最高值
            best_index = max(valid_indexes, key=lambda i: res_acc_list[i])
            #if task_type == 'search':
            best_action_acc = action_acc_list[best_index]
            best_res_acc = res_acc_list[best_index]

            # 统计最优 action_acc
            stats[task_type]["action_sum"] += best_action_acc
            global_stats["action_sum"] += best_action_acc

            # 统计与最优 action_acc 对应的 res_acc
            stats[task_type]["res_sum"] += best_res_acc
            global_stats["res_sum"] += best_res_acc
        else:
            # 如果没有 action_acc == 1 的情况，action_acc 和 res_acc 都为 0
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
            avg_interaction_times = interaction_count / total_count  # 计算平均交互次数
        else:
            avg_action_acc, avg_res_acc, avg_interaction_times = 0, 0, 0
        
        final_stats[task_type] = {
            "total_count": total_count,
            "avg_interaction_times": avg_interaction_times,  # 每种任务的平均交互次数
            "avg_action_acc": avg_action_acc,  # 平均 action_acc
            "avg_res_acc": avg_res_acc  # 平均 res_acc
        }
    
    global_total_count = global_stats["total_count"]
    global_interaction_count = global_stats["interaction_count"]
    if global_total_count > 0 and global_interaction_count > 0:
        global_avg_action_acc = global_stats["action_sum"] / global_total_count
        global_avg_res_acc = global_stats["res_sum"] / global_total_count
        global_avg_interaction_times = global_interaction_count / global_total_count  # 全局平均交互次数
    else:
        global_avg_action_acc, global_avg_res_acc, global_avg_interaction_times = 0, 0, 0
    
    final_stats["overall"] = {
        "total_count": global_total_count,
        "avg_interaction_times": global_avg_interaction_times,  # 全局的平均交互次数
        "avg_action_acc": global_avg_action_acc,
        "avg_res_acc": global_avg_res_acc
    }

    return final_stats





file_path = "results/step10_function_calling0-gpt-4o-mini-0.0_memrelevant_range0--1_usergpt-4o-mini_09181853.json"
res_file = json.load(open(file_path, "r"))
print(f'Showing results for{file_path}')
results = res_file
stats = calculate_statistics(results)
for task_type, stats in stats.items():
    print(f"\nTask type: {task_type}")
    for key, value in stats.items():
        print(f"{key}: {value}")
