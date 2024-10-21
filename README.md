# PersonalWAB: Large Language Models Empowered Personalized Web Agents

## Overview

This repository provides the implementation of **PUMA** (Personalized User Memory-enhanced Alignment) and the **PersonalWAB** (Personalized Web Agent Benchmark). The project addresses the limitations of current LLM-based Web agents by incorporating personalized data, such as user profiles and historical web behaviors, to improve user instruction understanding and customize actions.

The benchmark focuses on personalized Web agent tasks like personalized search, recommendation, and review generation.

We will maintain a **leaderboard** to allow researchers to submit and compare their models' performance on PersonalWAB. Details on how to participate and submit results will be released soon.

## Installation

### Requirements

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.5

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```


## PersonalWAB Benchmark

The **PersonalWAB** benchmark includes:

- **Personalized User Data**: 1,000 diverse user profiles and 40,000+ web behaviors, originated from real-world data.
- **User Instructions**: 9,000+ highly personalized natural language instructions tailored to each user's profile.
- **User Simulatior**: Simulates interactions aligned with user profiles and historical behaviors.
- **Evaluation Paradigms**:  Single-turn track tests for isolated tasks and multi-turn for more complex interactions.

The dataset is available in "PersonalWAB/envs/pwab/data".

### Task Description

**Personalized Search**: Personalized product search using user instructions and behavioral history.  
**Personalized Recommendation**: Recommend items based on implicit preferences.  
**Personalized Review Generation**: Generate reviews aligned with user preferences.


## Usage

### Running Experiments 

To run experiments on the **PersonalWAB** benchmark, use the following command:

```bash
source scripts/set_api_key.sh # Set your OpenAI API key
bash scripts/run_singleturn.sh  # Single-turn track
bash scripts/run_multiturn.sh   # Multi-turn track
```

You can modify agent strategies, memory mechanisms, and parameters in the scripts to explore various configurations.


## Benchmark Evaluation

The **PersonalWAB** benchmark supports two evaluation tracks: single-turn and multi-turn interactions. The key metrics for evaluation include:

- **Tool Accuracy**: The accuracy of selecting appropriate web functions.
- **Result Accuracy**: The relevance of returned results to user preferences.
- **Avg. Steps**: The average number of actions executed to complete a user instruction.

## PUMA Framework

The **PUMA** framework adapts LLMs for personalized Web agent tasks by utilizing:

- **Long-term Memory Bank**: A retrieval mechanism that filters relevant historical web behaviors.
- **Fine-tuning**: LLM fine-tuning with heuristically generated pseudo-labels.
- **Direct Preference Optimization**: Aligns parameter generation with user preferences through DPO.

For more detailed information, refer to our paper.

### Training PUMA

STEP 1: Prepare the SFT dataset  
```bash
cd PUMA
bash scripts/pre_sft_func_data.sh
bash scripts/pre_sft_param_data.sh
```
STEP 2: Train the LLaMA model with SFT  
```bash
bash scripts/finetune_function_param.sh
```
STEP 3: Generate function results and parameters for DPO  
```bash
bash scripts/generate_function.sh
bash scripts/generate_param_dpo.sh
```
STEP 4: Evaluate the parameter results in PersonalWAB  
```bash
cd ..
bash scripts/fast_test_dpo.sh
```
STEP 5: Prepare the DPO dataset  
```bash
cd PUMA
bash scripts/pre_dpo_data.sh
```
STEP 6: Train with DPO    
```bash
bash scripts/dpo_llama.sh
```
STEP 7: Evaluate the DPO model in PersonalWAB  
```bash
cd ..
bash scripts/run_singleturn_puma.sh
```
Or you can also generate the final function and parameter results and use scripts/fast_test.sh to see the performance without recording the single instruction results.

## Citation

If you use this code or dataset in your research, please cite our paper:


## Contact

