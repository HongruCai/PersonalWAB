deepspeed --master_port=29500 dpo_llama.py \
    --data_path output/res/ \
    --output_dir output/dpo \
    --model_name output/merged_model \
    --model_path output/input/Llama-2-7b-chat-hf/ \
    --train_epoch 5 \
    --learning_rate 5e-5 \
    --train_batch_size 1 \
    --source_length 1024 \
    --warmup_ratio 0.1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 10 \
    --deepseed_config config/llama_ds_config.json \
    --gradient_accumulation_steps 16 \
    --float16 \
    --train_on param \
    --prompt_length 768 \
    --beta 0.1 