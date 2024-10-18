from PersonalWAB.envs.base import PWABaseEnv
from PersonalWAB.envs.pwab.data import data
from PersonalWAB.envs.pwab.functions import functions


SYS_PROMPT_SINGLE = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- You are not allowed to interact with the user. Make the best tool call based on the user's request.
- You are only allowed to make one try. Once you make a tool call to search_product_by_query, get_recommendations_by_history or add_product_review, task will be over.
- The evaluation will be based on the tool selection and ranking of the target product in search and recommendation tasks, and the similarity of the review in the review task.
- If memory about the user is provided, you should use it to help you formulate better tool calls.
'''


SYS_PROMPT_MULTI = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- You are allowed to interact with the user to ask for more information or feedback, but total steps are limited to <NUM>, and less steps are preferred.
- Your main goal is to help the user complete the task as accurately and efficiently as possible, do not keep responding to the user, focus on making the best tool calls.
- When you think you have found the best input for the tool call, you can end the task by making a 'stop' call. 
- If the user wants to proceed to next step or express satisfaction, you should make a 'stop' call.
- The evaluation will be based on the tool selection and ranking of the target product in search and recommendation tasks, and the similarity of the review in the review task.
- You can make 'stop' call early, balance between the number of steps and the quality of the results.
- You should not make up any information or knowledge not provided from the user or the tools, or give subjective comments or suggestions.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
- If memory about the user is provided, you should use it to help you formulate better tool calls.
'''


class MockPWADomainEnv(PWABaseEnv):
    def __init__(
        self,
        user_mode: str = "naive",
        user_model: str = "gpt-4o-mini",
        task_split: str = "test",
        max_steps: int = -1,
    ):
        if max_steps == -1:
            sys_prompt = SYS_PROMPT_SINGLE
            tasks = data["tasks"][task_split]
        else:
            sys_prompt = SYS_PROMPT_MULTI.replace("<NUM>", str(max_steps))
            tasks = data["tasks"][task_split]


        super().__init__(data, functions, tasks, user_mode, user_model, sys_prompt, max_steps)
