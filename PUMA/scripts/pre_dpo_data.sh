python prepare_dpo_data.py \
    --task_file data/user_instructions.json \
    --user_history_file data/user_history.json \
    --dpo_data_file data/dpo_data.json \
    --output_file data/dpo_training_data.json \
    --mem_token_length 768 \
    --mem_length 100