python run.py \
--env pwab \
--model finetune/llama \
--user_mode no \
--user_model gpt-4o-mini \
--agent_strategy function_calling \
--agent_memory taskspe \
--memory_length 100 \
--task_split test \
--max_concurrency 5 \
--max_steps -1 \
--end_index -1 \
--puma_function_file PUMA/output/res/function_768.json \
