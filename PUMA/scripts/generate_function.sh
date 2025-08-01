accelerate launch test_llama.py \
    --model_path output/param/Llama-2-7b-chat-hf/xxx \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --data_path data/function_data.json \
    --bf16 \
    --test_on function \
    --batch_size 8 \
    --max_new_tokens 32 \
    --res_file output/res/function_res.json \
