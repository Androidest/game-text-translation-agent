# Using the Seq2Seq generative approach is better for term extraction tasks with a decoder-only model(auto-regressive model)
from peft import PeftModelForCausalLM, LoraConfig, PeftConfig
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from utils import *
from typing import Union, List
import torch
import json

# 
"""
Chat Template used by QWEN3 (without thinking)

For structured output (JSON list) generation:
    <|im_start|>system
    提取文中的游戏名词和术语，返回JSON列表<|im_end|>
    <|im_start|>user
    {user_input}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    ["

For teacher-forcing training,
input:
    <|im_start|>system
    提取文中的游戏名词和术语，返回JSON列表<|im_end|>
    <|im_start|>user
    {user_input}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

target:
    ["术语1", "术语2", "术语3" ... ]<|im_end|>
"""

class QwenGameTermTokenizer(Qwen2TokenizerFast):
    STRUCTURED_OUTPUT_PREFIX = '["'
    IGNORE_INDEX = -100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_term_extraction_template(
        self, 
        user_text:Union[str, List[str]], 
        target_text:Union[str, List[str]] = None,
    )->Union[str, List[str]]:
        
        has_target = target_text != None
        if has_target and type(user_text) != type(target_text):
            raise ValueError(f"The given 'user_text' & 'target_text' is type of {type(user_text)} & {type(target_text)}, target_text must be the same type as 'input_text' or None")

        if isinstance(user_text, str):
            user_text = [user_text]
        if not isinstance(user_text, list):
            raise ValueError(f"The given 'user_text' is type of {type(user_text)}, must be type of Union[str, List[str]]")
        if has_target:
            if isinstance(target_text, str):
                target_text = [target_text]
            if len(user_text) != len(target_text):
                raise ValueError("tThe given target_text and user_text are not the same length.")
              
        sys_prompt = { "role":"system", "content":"提取文中的游戏名词和术语，返回JSON列表" }
        requests = []
        responses = []
        for i in range(len(user_text)):
            conversation = [ 
                sys_prompt, 
                { "role":"user", "content":user_text[i] },
            ]
            prompts = self.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True,
                # continue_final_message=True, # True:do not end with <|endoftext|>，False：end with <|endoftext|>
            )
            if has_target:
                responses.append(target_text[i] + '<|im_end|>')
            else:
                prompts += self.STRUCTURED_OUTPUT_PREFIX

            requests.append(prompts)

        if has_target:
            return requests, responses
        return requests

    def get_terms_from_output(self, output_text:Union[str, List[str]], return_list:bool=True):
        if isinstance(output_text, list):
            return [ self._get_terms_from_output(t, return_list) for t in output_text]
        elif isinstance(output_text, str):
            return self._get_terms_from_output(output_text, return_list)
    
    def _get_terms_from_output(self, output_text:str, return_list:bool=True):
        start = output_text.find('[')
        if start == -1:
            start = 0
            output_text = self.STRUCTURED_OUTPUT_PREFIX + output_text
        
        end = output_text.find(']', start)
        if end == -1:
            return None
        
        json_list = output_text[start:end+1]
        if not return_list:
            return json_list

        try:
            return json.loads(json_list)
        except:
            return None
    
class QwenGameTermLoraModel(PeftModelForCausalLM):
    def __init__(self, model:torch.nn.Module, peft_config:PeftConfig=None, adapter_name:str='default', **kwargs):
        if peft_config is None:
            peft_config = LoraConfig(
                r=8, # rank
                lora_alpha=4, # controls the multiplier α/r in ΔW=r/α*​AB
                lora_dropout=0.07,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
            )
        super().__init__(model=model, peft_config=peft_config, adapter_name=adapter_name, **kwargs)

if __name__ == "__main__":
    MODEL_PATH = get_model_local_path(ModelID.QWEN3)
    DS_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'

    tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(MODEL_PATH)
    sheet = Sheet(DS_SHEET_PATH)[41118:41118+3]
    cn = list(sheet['CN'])
    terms = list(sheet['TERMS'])

    print("Inputs for Training:")
    mock_outputs = []
    requests, responses = tokenizer.apply_term_extraction_template(user_text=cn, target_text=terms)
    for req, res in zip(requests, responses):
        train_input = req + res
        mock_outputs.append(res)
        print(train_input.replace('\n', '\\n'))
        print('-'*50)

    print("Test terms extraction on generated texts:")
    extracted_terms = tokenizer.get_terms_from_output(mock_outputs, return_list=True)
    for e, t in zip(extracted_terms, terms):
        print('原有的术语', t)
        print('抽取的术语', e)
        print('-'*50)

    print("Inputs for Evaluation:")
    requests = tokenizer.apply_term_extraction_template(user_text=cn)
    for req in requests:
        print(req.replace('\n', '\\n'))
        print('-'*50)
