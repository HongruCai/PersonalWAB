deepspeed --master_port=23333 finetune_llama.py \
    --data_path data/param_data768.json \
    --function_data_path data/function_data1.json \
    --output_dir output/param \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --train_epoch 10 \
    --learning_rate 3e-4 \
    --train_batch_size 1 \
    --source_length 1024 \
    --warmup_ratio 0.1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 10 \
    --deepseed_config config/llama_ds_config.json \
    --gradient_accumulation_steps 16 \
    --temperature 1.0 \
    --float16 \
    --train_on function_param \