import os
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding, GenerationConfig
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
    logging,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import random
import wandb
# import deepspeed
from typing import Dict, List
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
import argparse
import sys
from utils import LlaMaTrainerwithTemperature, LLaMaDataset
from torch.utils.data import ConcatDataset
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llama")

    parser.add_argument('--data_path', type=str, default='data', help='param data path')
    parser.add_argument('--function_data_path', type=str, default='data/function_data.json', help='function data path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--model_name', type=str, default='llama-2-hf', help='model name')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--wandb_log_freq', type=int, default=5, help='wandb log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--train_on', type=str, default='tool', choices=['function', 'param','function_param'], help='train on function or param or both')
    
    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path
    function_data_path = train_args.function_data_path

    print('training on: ', data_path)

    model_name = train_args.model_name
    
    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    source_length = train_args.source_length
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = current_time+'_'+str(data_path.split('/')[-1])+'_ep'+str(train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)

    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] + '/' + output_dir
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if ddp:
        device_map = {"": local_rank}
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch_dtype,
                                             config=config,
                                            #load_in_8bit=True,
                                            #load_in_4bit=True,
                                            #bnb_4bit_compute_dtype=torch_dtype,
                                             device_map=device_map)

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        modules_to_save=['embed_tokens', 'lm_head',
                         'input_layernorm', 'post_attention_layernorm', 'norm'],
        target_modules=['q_proj', 'v_proj', 'k_proj',
                        'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
        lora_dropout=0,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    reporter = ['wandb'] if local_rank == 0 else "none"

    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,         
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=train_batch_size, 
        dataloader_num_workers=10,

        #optim = "adamw_8bit",   
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        # weight_decay=0.01,   
                   
        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        evaluation_strategy=train_args.eval_strategy,
        save_strategy=train_args.save_strategy,
        save_total_limit=train_args.save_total_limit,

        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        fp16=train_args.float16,
        bf16=train_args.bf16,

        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        save_only_model=True,
    )


    if train_args.train_on == 'function_param':
        param_train_dataset = LLaMaDataset(tokenizer, json_file=data_path, max_length=source_length, split="train")
        param_test_dataset = LLaMaDataset(tokenizer, json_file=data_path, max_length=source_length, split="test")
        function_train_dataset = LLaMaDataset(tokenizer, json_file=function_data_path, max_length=source_length, split="train")
        function_test_dataset = LLaMaDataset(tokenizer, json_file=function_data_path, max_length=source_length, split="test")
        train_dataset = ConcatDataset([param_train_dataset, function_train_dataset])
        test_dataset = ConcatDataset([param_test_dataset, function_test_dataset])
    elif train_args.train_on == 'param':
        train_dataset = LLaMaDataset(tokenizer, json_file=data_path, max_length=source_length, split="train")
        test_dataset = LLaMaDataset(tokenizer, json_file=data_path, max_length=source_length, split="test")
    elif train_args.train_on == 'function':
        train_dataset = LLaMaDataset(tokenizer, json_file=function_data_path, max_length=source_length, split="train")
        test_dataset = LLaMaDataset(tokenizer, json_file=function_data_path, max_length=source_length, split="test")


    print('training dataset size: ', len(train_dataset))
    print('test dataset size: ', len(test_dataset))
    

    if local_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
        logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)

        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transfomers training_args: '+str(training_args))


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length, return_tensors="pt")
    trainer = LlaMaTrainerwithTemperature(
        temperature=train_args.temperature,
        vocab_size=len(tokenizer),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    trainer.train()

