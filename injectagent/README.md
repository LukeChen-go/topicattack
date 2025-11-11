### Agent Evaluation

We have crafted the dataset with **TopicAttack** and you can find them in `data/crafted_instruction_data_agent_grant.json`.
To evaluate the performance, you can run the following code :

```angular2html
cd src

python evaluate_prompted_agent.py \
    --model_type GPT \
    --model_name gpt-4o \
    --setting base \
    --prompt_type InjecAgent \
    --attacks conv_attack \
    --defenses sandwich none  spotlight \
    --test_case_file "../data/crafted_instruction_data_agent_grant.json" \
    --log_file "logs/conv-grant-gpt-4o.txt"
```
You can also replace the victim model with other models.

### Attack Construction

To construct the attack data, please put your `OpenAI api key` into the `chatbot.py`. Then you can run the following code:

```angular2html
cd src

python construct_attack.py \
    --model gpt-4o \
    --data_path ../data/test_cases_dh_base.json \
    --template_path ../../prompts/attack_prompt_transfer_agent.txt \
    --chatbot_system_path ../../prompts/chatbot_system.txt \
    --output_path  ../data/crafted_instruction_data_agent_grant.json
```