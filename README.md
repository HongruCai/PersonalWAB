# PersonalWAB: Large Language Models Empowered Personalized Web Agents

## Overview

This repository provides the implementation of **PUMA** (Personalized User Memory-enhanced Alignment) and the **PersonalWAB** (Personalized Web Agent Benchmark). This project addresses the limitations of existing Large Language Model (LLM)-based Web agents by incorporating personalized data (e.g., user profiles, historical web behaviors) to enhance the understanding of user instructions and customize action execution.

## Installation

### Requirements

- Python 3.9
- PyTorch
- [List other libraries or tools your project depends on]

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

## PersonalWAB Benchmark

The **PersonalWAB** benchmark includes:
- **User Instructions**: Natural language instructions for Web task completion.
- **Personalized User Data**: User-specific data (profiles, historical behaviors).
- **Web Functions**: Actions that agents can perform based on instructions.
- **Evaluation Paradigms**: Two paradigms across three personalized Web tasks.

The dataset has been contained in "PersonalWAB/envs/pwa/data".

---

## Usage

### Running Experiments 



---

## Benchmark Evaluation

The **PersonalWAB** benchmark provides two evaluation paradigms across three tasks:
1. **Task 1**: [Description of task]
2. **Task 2**: [Description of task]
3. **Task 3**: [Description of task]

### Benchmark Results

We provide the following metrics for evaluation:


---

## PUMA Framework

The **PUMA** framework adapts LLMs for personalized Web agent tasks by utilizing:
- **Memory Bank**: A task-specific retrieval mechanism that filters relevant historical Web behaviors.
- **Fine-tuning**: LLM fine-tuning with heuristically generated pseudo-labels.
- **Direct Preference Optimization**: A strategy to align model predictions with user preferences.

For more details, please refer to our paper.

### Training PUMA

STEP 1: Prepare the dataset
```bash
cd PUMA
bash scripts/pre_sft_func_data.sh
bash scripts/pre_sft_param_data.sh
```
STEP 2: Train the LLaMA model with SFT
```bash
bash scripts/fintune_function_param.sh
```
STEP 3: Generate function results and parameter results for DPO
```bash
bash scripts/genenrate_function.sh
bash scripts/genenrate_param_dpo.sh
```
STEP 4: Evaluate the parameter results in PersonalWAB
```bash
cd ..

---

## Citation

If you use this code or dataset in your work, please cite our paper:

---

## Contact

