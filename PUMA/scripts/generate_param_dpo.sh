accelerate launch test_llama.py \
    --model_path output/dpo/merged_model/ \
    --base_model output/merged_model \
    --data_path data/tool_input_data3.json \
    --float16 \
    --test_on param \
    --batch_size 4 \
    --max_new_tokens 512 \
    --tool_file output/ \
    --res_file output/