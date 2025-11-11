# Large Language Models Empowered Personalized Web Agents

## 🔍 Overview

LLM-based Web agents overlook the importance of personalized data (e.g., user profiles and historical Web behaviors) in assisting the understanding of users' personalized instructions and executing customized actions. **PersonalWAB** (Personalized Web Agent Benchmark) serves as the first comprehensive benchmark designed to evaluate Web agents on tasks such as personalized search, recommendation, and review generation. The benchmark includes a set of personalized user data, Web functions, and evaluation paradigms that facilitate the development of more personalized Web agents.
**PUMA** (Personalized User Memory-enhanced Alignment) is a framework developed to adapt LLMs to the personalized Web agent task. By leveraging a memory bank and task-specific retrieval strategies, PUMA filters relevant historical Web behaviors, enabling fine-tuned and optimized personalized action execution.


> For more details, refer to our paper accepted to **WWW 2025**: [Large Language Models Empowered Personalized Web Agents](https://arxiv.org/abs/2410.17236).

## ⚙️ Installation

### Requirements

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.5
- openjdk

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```


## 📊 PersonalWAB Benchmark

![](https://hongrucai.github.io/images/personalwab.png)

The **PersonalWAB** benchmark includes:

- **Personalized User Data**: 1,000 diverse user profiles and 40,000+ web behaviors, originated from real-world data.
- **User Instructions**: 9,000+ highly personalized natural language instructions tailored to each user's profile.
- **User Simulatior**: Simulates interactions aligned with user profiles and historical behaviors.
- **Evaluation Paradigms**:  Single-turn track tests for isolated tasks and multi-turn for more complex interactions.

The dataset is available in "PersonalWAB/envs/pwab/data". Or you can download [here](https://hongrucai.github.io/PersonalWAB/download).

### Task Description

**Personalized Search**: Personalized product search using user instructions and behavioral history.  
**Personalized Recommendation**: Recommend items based on implicit preferences.  
**Personalized Review Generation**: Generate reviews aligned with user preferences.

### Running Experiments 

To run experiments on the **PersonalWAB** benchmark, use the following command:

```bash
source scripts/set_api_key.sh  # Set your OpenAI API key
bash scripts/run_singleturn.sh  # Single-turn track
bash scripts/run_multiturn.sh   # Multi-turn track
```

You can modify agent strategies, memory mechanisms, and parameters in the scripts to explore various configurations.

For experiments using task-specific memory, use the PUMA framework to generate function selection results. For InteRecAgent, you need to provide the memory file generated from history behaviors before running in training or test set.

### Benchmark Evaluation

The **PersonalWAB** benchmark supports two evaluation tracks: single-turn and multi-turn interactions. The key metrics for evaluation include:

- **Function Accuracy**: The accuracy of selecting appropriate web functions.
- **Result Accuracy**: The relevance of returned results to user preferences.
- **Avg. Steps**: The average number of actions executed to complete a user instruction.

### Leaderboard Submission

When running `run.py`, the script will automatically save the detailed task execution results. This includes the agent's prompts, actions, function accuracy (Function Acc), result accuracy (Res Acc), and other relevant information.

- After testing, the program will generate a detailed result file with all the execution data.
- Upload this complete result file to an online storage service such as Google Drive or OneDrive.
- Once uploaded, please submit the download link through the provided [Google Form](https://forms.gle/UQdxUG8f28xbRd5Z8).
- Ensure that the result file is accessible via the shared link, and that it contains all relevant information for evaluation and ranking on the leaderboard.


## 🧠 PUMA Framework

The **PUMA** framework adapts LLMs for personalized Web agent tasks by utilizing:

- **Long-term Memory Bank**: A retrieval mechanism that filters relevant historical web behaviors.
- **Fine-tuning**: LLM fine-tuning with heuristically generated pseudo-labels.
- **Direct Preference Optimization**: Aligns parameter generation with user preferences through DPO.

For more detailed information, refer to our paper.

### Training PUMA

**STEP 1: Prepare the SFT dataset.**

You may need to copy the 'user_interactions.json' and 'user_history.json' files from the PersonalWAB benchmark.    
```bash
cd PUMA
bash scripts/pre_sft_func_data.sh
bash scripts/pre_sft_param_data.sh
```
**STEP 2: Train the LLaMA model with SFT**  

There are three options for training: function, parameter, or both, which means SFT on function data, parameter data, or both function and parameter. You can specify in the script.
```bash
bash scripts/finetune_function_param.sh
```
**STEP 3: Generate function results and diverse parameter predictions for DPO** 

This step generates the function predictions and diverse parameter predictions for DPO training. The function results will be used to select the appropriate Web functions, while the parameter results are parameters for the selected functions. You may need to generate on training and test set separately by specifying the split.
```bash
bash scripts/generate_function.sh
bash scripts/generate_param_dpo.sh
```
**STEP 4: Evaluate the parameter results in PersonalWAB**  

This step evaluates the parameter results in the PersonalWAB benchmark. It will score the parameter results in the PersonalWAB benchmark, which will be used in the DPO training.
```bash
cd ..
bash scripts/fast_test_dpo.sh
```
**STEP 5: Prepare the DPO dataset**  

This step prepares the DPO dataset by choosing the highest and lowest parameter results from the previous step. 
```bash
cd PUMA
bash scripts/pre_dpo_data.sh
```
**STEP 6: Train with DPO**  

This step trains the model with DPO using the prepared DPO dataset. You need to first merge the LoRA adapter saved in SFT with base model, and then train the DPO model with the merged model.
```bash
bash scripts/merge.sh
bash scripts/dpo_llama.sh
```
**STEP 7: Evaluate the DPO model in PersonalWAB**  
This step evaluates the final DPO model in the PersonalWAB benchmark. But you can also use the model to generate the final function and parameter results (as in step 3)and use scripts/fast_test.sh to see the performance.
```bash
cd ..
bash scripts/run_singleturn_puma.sh
```

## 📚 Citation

If you use source code or dataset in your research, please cite our paper:
```bibtex
@inproceedings{cai2024personalwab,
      title={Large Language Models Empowered Personalized Web Agents}, 
      author={Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li, Tat-Seng Chua},
      year={2025},
      booktitle={Proceedings of the ACM Web Conference 2025},
      series={WWW'25}
}
```

## 📄 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

The benchmark implementation in this project is based on the [tau-bench](https://github.com/sierra-research/tau-bench), with significant modifications and enhancements made to suit the needs of this project. The tau-bench is originally licensed under the [MIT License](https://github.com/sierra-research/tau-bench?tab=MIT-1-ov-file), and we continue to honor and adhere to its licensing terms for the portions derived from it.

## 📬 Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).
