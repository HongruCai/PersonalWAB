from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
from utils import load_param_prompt, load_function_prompt
import json
from accelerate import PartialState
from accelerate.utils import gather_object

def parse_args():
    parser = argparse.ArgumentParser(description="Test Llama model")

    parser.add_argument('--model_path', type=str, default='output/', help='model path')
    parser.add_argument('--data_path', type=str, default='data/', help='data path')
    parser.add_argument('--history_path', type=str, default='data/user_history.json', help='history path')
    parser.add_argument('--device', type=str, default='cuda', help='device') 
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='base model')
    parser.add_argument('--sample_num', type=int, default=None, help='number of samples')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams')
    parser.add_argument('--split', type=str, default='test', help='split to evaluate')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--test_on', type=str, default='function', choices=['function', 'param', 'function_param'], help='test on tool or input')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max new tokens')
    parser.add_argument('--memory_token_length', type=int, default=768, help='memory token length')
    parser.add_argument('--tool_file', type=str, default='data/', help='task file')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--res_file', type=str, default='output/', help='result file')
    parser.add_argument('--temperature', type=float, default=0, help='temperature')
    parser.add_argument('--do_sample', action='store_true', help='do sample')
    return parser.parse_args()


def batch_inference(model, tokenizer, inputs, labels, batch_size, device, args, max_new_tokens):
    rec = {'search': [], 'rec': [], 'review': []}

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=max_new_tokens,
        num_return_sequences=args.num_beams,
        early_stopping=True if args.num_beams != 1 else False,
        use_cache=True,
        
        temperature=args.temperature if args.temperature > 0 else 0,
        do_sample=args.do_sample,
    )

    model.eval()  
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc='Evaluating batches'):
            batch_inputs = inputs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            tokenized_prompts = [
                tokenizer(input_text, return_tensors="pt").input_ids
                for input_text in batch_inputs
            ]

            res = []
            with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
                for batch in batched_prompts:
                    batch = batch.to(distributed_state.device)

                    beams = model.generate(batch, generation_config=generation_config)
                    #print(tokenizer.decode(beams[0], skip_special_tokens=True).strip())

                    if args.test_on == 'function':
                        beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('Tool:\n')[-1] for x in beams]
                    elif args.test_on == 'param':
                        beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('### Tool Input:\n')[-1] for x in beams]

                    res.extend(beams)
            res = gather_object(res)

            for j in range(len(batch_labels)):
                label = batch_labels[j]
                result[tasks[i+j]] = res[j*args.num_beams:(j+1)*args.num_beams]

    return rec


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)
    print(device)

    base_model = args.base_model
    model_path = args.model_path

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    distributed_state = PartialState()

    if args.float16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )
    model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch_dtype,
            device_map=distributed_state.device
        )


    print('tokenizer loaded from '+model_path)
    print('model loaded from '+model_path)

    model.to(device)
    model.eval()

    data_path = args.data_path
    batch_size = args.batch_size

    sample_num = args.sample_num
    num_beams = args.num_beams

    valid_modes = args.split.split(',')

    rec = {'search':[], 'rec':[], 'review':[]}
    result = {}
    for valid_mode in valid_modes:

        if args.test_on == 'function':
            tasks, total_inputs, total_labels = load_function_prompt(args.data_path, valid_mode)
        elif args.test_on == 'param':
            tasks, total_inputs, total_labels = load_param_prompt(args.data_path, args.tool_file, valid_mode, args.memory_token_length, tokenizer)

        print('Data loaded from '+data_path)
        rec = batch_inference(model, tokenizer, total_inputs, total_labels, batch_size, device, args, args.max_new_tokens)

    with open(args.res_file, 'w') as f:
        json.dump(result, f, indent=2)
                

          

        