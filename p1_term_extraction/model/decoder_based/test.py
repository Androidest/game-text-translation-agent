from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, Qwen3ForCausalLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, get_peft_model, LoraConfig, TaskType, PeftModel
from utils import *
import torch
import pdb

SYS_PROMPT = "提取文中的游戏名词和术语，返回JSON列表"

PROMPT = """\
<|im_start|>system
{sys_propmt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
[\""""

NO_THINKING_PROMPT = """\
<|im_start|>system
{sys_propmt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
[\""""

def save_lora(base_model, lora_path):
    peft_config = LoraConfig(
        r=64, # rank
        lora_alpha=16, # controls the multiplier α/r in ΔW=r/α*​AB
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM, # to warp the model into a PeftModelForCausalLM (overriding the forward() needs to recompute the loss etc.)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
    )
    lora_model = get_peft_model(model=base_model, peft_config=peft_config)
    lora_model.save_pretrained(lora_path)

def load_lora(model, lora_path):
    lora_model:PeftModelForCausalLM = PeftModelForCausalLM.from_pretrained(
        model=model,
        model_id=lora_path,
        adapter_name="extractor",
        is_trainable=False,
    )
    lora_model.set_adapter("extractor")
    return lora_model

@torch.no_grad()
def extract_terms(user_input, model, tokenizer, device):
    # text = tokenizer.apply_chat_template(
    #     conversation = [
    #         {"role": "user", "content": user_input}
    #     ],
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     enable_thinking=False
    # )
    
    text = NO_THINKING_PROMPT.format(sys_propmt=SYS_PROMPT, user_input=user_input)
    x = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    x = {k:v.to(device) for k,v in x.items()}

    gen_config = model.generation_config
    gen_config.max_new_tokens = 1024
    gen_config.temperature = 0.001
    gen_config.top_p = 0.1
    gen_config.repetition_penalty = 1.0

    y = model.generate(**x, generation_config=gen_config)
    y = tokenizer.decode(y[0], skip_special_tokens=False)
    return y

if __name__ == "__main__":
    MODEL_PATH = get_model_local_path(ModelID.QWEN3)
    MODEL_LORA_PATH = str(get_model_local_path(ModelID.QWEN3_LORA, ModelSrc.LOCAL))
    DEVICE = 'cuda:0'

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model Type:", type(model))
    print("Tokenizer Type:", type(tokenizer))

    save_lora(model, MODEL_LORA_PATH)
    model = load_lora(model, MODEL_LORA_PATH)
    model.eval()
    model.to(DEVICE)

    user_inputs = [
        "角色攻击力+15%，穿透率+15%，无视格挡+12%，+5%暴击率，对所有属性怪物增加16%伤害（可与其他回响叠加）",
        "角色暴击率+10%，暴击伤害+35%，无视格挡+8%，对弱火、无属性怪物增加60%伤害",
        "凤眼蓝使用技能后，立即减少另一个花灵25%正在冷却中的花灵技能剩余冷却时间，且下一个花灵造成的下一次技能伤害提升75%",
    ]
    
    for user_input in user_inputs:
        print(extract_terms(user_input, model, tokenizer, DEVICE))
        print('-'*50) 