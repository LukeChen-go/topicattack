# TopicAttack: An Indirect Prompt Injection Attack via Topic Transition

Official code implementation for EMNLP 2025 accepted paper: TopicAttack: An Indirect Prompt Injection Attack via Topic Transition (https://arxiv.org/pdf/2507.13686)

### Environment
```
conda creat -n topicattack python=3.10
conda activate topicattack
pip install openai
```

### Chatbot Evaluation

We have crafted the dataset with **TopicAttack** and you can find them in `data/crafted_instruction_data_squad_conversation_attack_complete.json` and `data/crafted_instruction_data_tri_conversation_attack_complete.json`.
To evaluate the performance, you can run the following code :

```angular2html
python run_baselines.py \
    --victim_model_path Qwen/Qwen2-7B-Instruct \
    --data_path data/crafted_instruction_data_squad_conversation_attack_complete.json \
    --victim_system_path prompts/generator_system_prompt.txt \
    --attacks conv_attack \
    --defenses none sandwich  \
    --log_file squad_logs/sandwich-conv-ignore-qwen2-7b-squad-complete.txt
```
You can also replace the victim model with other models such as `GPT-4o`.

### Attack Construction

To construct the attack data, please put your `OpenAI api key` into the `chatbot.py`. Then you can run the following code:

```angular2html
python construct_attack.py \
    --model gpt-4o \
    --data_path ./data/crafted_instruction_data_squad_injection_qa.json \
    --template_path prompts/attack_prompt_transfer.txt \
    --chatbot_system_path prompts/chatbot_system.txt \
    --output_path  ./data/crafted_instruction_data_squad_conversation_attack_complete.json
```

### Agent Evaluation

Please step into the `injectagent` directory for more details, we implement our attack based on [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent).
