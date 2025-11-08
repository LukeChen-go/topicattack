import os
from src.utils import get_response_text
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class ClaudeModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(
            api_key= os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.params = params
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"{self.human_prompt} {sys_prompt} {user_prompt_filled}{self.ai_prompt}"
        return model_input

    def call_model(self, model_input):
        completion = self.anthropic.completions.create(
            model=self.params['model_name'],
            max_tokens_to_sample=4096,
            prompt=model_input,
            temperature=0
        )
        return completion.completion


class HuggingfaceModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = self.load_hugging_face_model(self.params['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.params['model_name'])


    def load_hugging_face_model(self, model, max_mem_per_gpu='80GiB'):
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

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):

        input_ids = self.tokenizer.apply_chat_template(
            model_input,
            tokenize=True,
            add_generation_prompt=True
        )
        # input_ids = self.tokenizer(message).input_ids
        input_ids = torch.tensor(input_ids).view(1,-1).to(self.model.device)
        generation_config = self.model.generation_config
        generation_config.max_length = 8192
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




class GPTModel(BaseModel):
    def __init__(self, params):
        super().__init__()  
        from openai import OpenAI
        self.client = OpenAI(
            api_key="API_KEY"
            # organization = os.environ.get("OPENAI_ORGANIZATION")
        )
        self.params = params

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            temperature=0
        )
        return completion.choices[0].message.content

class APIModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.provider = {
            "meta-llama/llama-3.3-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/sambanova", # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "SambaNova", # Optional. Site title for rankings on openrouter.ai.
                },
            "deepseek/deepseek-chat":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/together",
                    "X-Title": "Together",
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
            "anthropic/claude-3.5-haiku":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/anthropic",
                    "X-Title": "Anthropic",
                },
            "qwen/qwen-2-72b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/together",
                    "X-Title": "Together",  # Optional. Site title for rankings on openrouter.ai.
                },
        }
        from openai import OpenAI
        self.client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key="API_KEY",
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers=self.provider[self.params['model_name']],
                    extra_body={},
                    model=self.params['model_name'],
                    messages=model_input,
                    temperature=0
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(e)
        return "fail"


    
import together   
class TogetherAIModel(BaseModel): 
    def __init__(self, params):
        super().__init__()  
        
        from src.prompts.prompt_template import PROMPT_TEMPLATE
        
        self.params = params
        self.model = self.params['model_name']
        self.prompt = PROMPT_TEMPLATE[self.model]

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = self.prompt.format(sys_prompt = sys_prompt, user_prompt = user_prompt_filled)
        return model_input
    
    def call_model(self, model_input, retries=3, delay=2, max_tokens =512):
        attempt = 0
        while attempt < retries:
            try:
                completion = together.Complete.create(
                    model=self.model,
                    prompt=model_input,
                    max_tokens=max_tokens,
                    temperature=0
                )
                return completion['choices'][0]['text']
            except Exception as e:
                attempt += 1
                max_tokens = max_tokens // 2
                print(f"Attempt {attempt}: An error occurred - {e}")
                if attempt < retries:
                    time.sleep(delay)  # Wait for a few seconds before retrying
                else:
                    return ""
    
class LlamaModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        import transformers
        import torch
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=4096,
            do_sample=False, # temperature=0
            eos_token_id=tokenizer.eos_token_id,
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"
        return model_input

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, "[/INST]")
    
MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel,
    "API": APIModel,
    "HF": HuggingfaceModel
}   