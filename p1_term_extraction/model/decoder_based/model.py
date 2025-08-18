from peft import PeftModelForCausalLM, LoraConfig, PeftConfig
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from utils import *
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

    def create_term_extraction_prompt(self, input_text:str, label_terms:str = None):
        # return self.final_prompt_template.format(user_input=text)
        input_text = self.apply_chat_template(
            conversation = [
                { "role":"system", "content":"提取文中的游戏名词和术语，返回JSON列表" },
                { "role":"user", "content":input_text }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if label_terms is None:
            input_text += '["'
        else:
            input_text += label_terms

        return input_text
    
    def get_terms_from_output(self, output_text:str, return_list:bool=True):
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