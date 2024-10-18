python prepare_dpo_data.py \
    --task_file data/user_instructions.json \
    --user_history_file data/user_history.json \
    --dpo_data_file output/res/dpo_data_256.json \
    --output_file data/param_dpo_data.json \
    --mem_token_length 768 \
    --mem_length 100