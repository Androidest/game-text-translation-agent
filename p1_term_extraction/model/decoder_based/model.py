# Using the Seq2Seq generative approach is better for term extraction tasks with a decoder-only model(auto-regressive model)
from peft import PeftModelForCausalLM, LoraConfig, PeftConfig
from transformers import Qwen2TokenizerFast
from utils import *
from typing import Union, List
import torch
import json

# Used by QWEN3
THINK_SPACEHOLDER = """
<think>

</think>"""

PROMPT_TEMPLATE = """\
<|im_start|>system
提取文中的游戏名词和术语，返回JSON列表<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant{think_spaceholder}
[\""""

class QwenGameTermTokenizer(Qwen2TokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_term_extraction_prompt(
        self, 
        input_text:Union[str, List[str]], 
        target_text:Union[str, List[str]] = None,
    )->Union[str, List[str]]:

        sys_prompt = { "role":"system", "content":"提取文中的游戏名词和术语，返回JSON列表" }
        continue_final_message = target_text == None

        if isinstance(input_text, list):
            if target_text == None:
                target_text = ['["'] * len(input_text)

            if not isinstance(target_text, list):
                raise ValueError(f"Given 'target_text' is type of {type(target_text)}, must be the same type as 'input_text' or None")

            conversation = [
                [ 
                    sys_prompt, 
                    { "role":"user", "content":user_text },
                    { "role":"assistant", "content":assistant_text },
                ]
                for user_text, assistant_text in zip(input_text, target_text)
            ]

        elif isinstance(input_text, str):
            if target_text == None:
                target_text = '["'

            if not isinstance(target_text, str):
                raise ValueError(f"Given 'target_text' is type of {type(target_text)}, must be the same type as 'input_text' or None")
            
            conversation = [ 
                sys_prompt, 
                { "role":"user", "content":input_text },
                { "role":"assistant", "content":target_text },
            ]
        else:
            raise ValueError(f"Given 'input_text' is type of {type(input_text)}, must be type of Union[str, List[str]]")

        # return self.final_prompt_template.format(user_input=text)
        return self.apply_chat_template(
            conversation = conversation,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=continue_final_message, # True:do not end with <|im_end|>（eval），False：end with <|im_end|>（train）
            enable_thinking=False,
        )
    
    def get_terms_from_output(self, output_text:Union[str, List[str]], return_list:bool=True):
        if isinstance(output_text, list):
            return [ self._get_terms_from_output(t, return_list) for t in output_text]
        elif isinstance(output_text, str):
            return self._get_terms_from_output(output_text, return_list)
    
    def _get_terms_from_output(self, output_text:str, return_list:bool=True):
        start = output_text.find('[')
        if start == -1:
            return None
        
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
                r=64, # rank
                lora_alpha=16, # controls the multiplier α/r in ΔW=r/α*​AB
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=True,
            )
        super().__init__(model=model, peft_config=peft_config, adapter_name=adapter_name, **kwargs)

if __name__ == "__main__":
    MODEL_PATH = get_model_local_path(ModelID.QWEN3)
    DS_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'

    tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(MODEL_PATH)
    sheet = Sheet(DS_SHEET_PATH)[41118:41118+3]
    cn = list(sheet['CN'])
    terms = list(sheet['TERMS'])

    samples = tokenizer.create_term_extraction_prompt(
        input_text=cn,
        target_text=terms,
    )

    for t in samples:
        print(t.replace('\n', '\\n'))

    extracted_terms = tokenizer.get_terms_from_output(samples)

    for e, t in zip(extracted_terms, terms):
        print('原有的术语', t)
        print('抽取的术语', e)
        print('-'*50)
