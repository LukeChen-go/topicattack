import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pprint
from openai import OpenAI
# openai.api_key =
from utils import load_text
import datetime



class HuggingfaceChatbot:
    def __init__(self, model, system_prompt, max_mem_per_gpu='40GiB'):
        self.model = self.load_hugging_face_model(model, max_mem_per_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.system_prompt = load_text(system_prompt)

    def load_hugging_face_model(self, model, max_mem_per_gpu='40GiB'):
        MAX_MEM_PER_GPU = max_mem_per_gpu
        map_list = {}
        for i in range(torch.cuda.device_count()):
            map_list[i] = MAX_MEM_PER_GPU
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            max_memory=map_list,
            torch_dtype="auto"
        )
        return model

    def respond(self, message, max_new_tokens=256, defense_cross_prompt=False):
        # global  SYS_INPUT
        if isinstance(message, list):
            messages = [{"role":"system", "content": self.system_prompt}] + message
        else:
            messages = [
                {"role":"system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        # input_ids = self.tokenizer(message).input_ids
        input_ids = torch.tensor(input_ids).view(1,-1).to(self.model.device)
        generation_config = self.model.generation_config
        generation_config.max_length = 8192
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = False
        # generation_config.do_sample = True
        generation_config.temperature = 0.0
        output = self.model.generate(
            input_ids,
            generation_config=generation_config
        )
        response = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        response = response.strip()
        return response

class GPTChatbot:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = load_text(system_prompt)
    def respond(self, message, max_new_tokens=256, seed=42):
        if isinstance(message, list):
            messages = [{"role":"system", "content": self.system_prompt}] + message
        else:
            messages = [
                {"role":"system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
        client = OpenAI(
            api_key="API_KEY",  # This is the default and can be omitted
        )
        # time.sleep(1)
        for _ in range(10):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_new_tokens,
                    n=1,
                    temperature=0.0,
                    seed=seed
                ).choices[0].message.content

                response = response.strip()
                return response
            except Exception as e:
                print(e)

        return "fail"



class OpensourceAPIChatbot:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = load_text(system_prompt)
        self.provider = {
            "meta-llama/llama-3.3-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/sambanova", # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "SambaNova", # Optional. Site title for rankings on openrouter.ai.
                },
            "qwen/qwen-2-72b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/together",
                    "X-Title": "Together",  # Optional. Site title for rankings on openrouter.ai.
                },
            "meta-llama/llama-3-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",  # Optional. Site title for rankings on openrouter.ai.
                },
            "meta-llama/llama-3.1-405b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",
                },
            "meta-llama/llama-3.1-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/sambanova",
                    "X-Title": "SambaNova",
                },
            "google/gemma-3-27b-it":{
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",
            },
            "qwen/qwen3-32b":{
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",
            }




        }

    def respond(self, message, max_new_tokens=256, seed=42):
        # global  SYS_INPUT
        if isinstance(message, list):
            messages = [{"role":"system", "content": self.system_prompt}] + message
        else:
            messages = [
                # {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
                {"role":"system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ]
        client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key="API_KEY",
        )
        # time.sleep(1)
        for _ in range(10):
            try:
                response = client.chat.completions.create(
                  extra_headers=self.provider[self.model],
                  extra_body={},
                  model=self.model,
                  messages=messages,
                  temperature=0.0,
                  max_tokens=max_new_tokens,
                  seed=seed
                ).choices[0].message.content
                return response
            except Exception as e:
                print(e)
        return "Fail"
