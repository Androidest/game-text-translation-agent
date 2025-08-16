from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, Qwen3ForTokenClassification, AutoModelForCausalLM, AutoTokenizer
from utils import *
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

if __name__ == "__main__":
    from modelscope.hub.snapshot_download import snapshot_download
    MODEL_PATH = get_llm_local_path(ModelID.QWEN3)
    DEVICE = 'cuda:0'

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model Type:", type(model))
    print("Tokenizer Type:", type(tokenizer))
    model.to(DEVICE)

    gen_config = model.generation_config
    gen_config.max_new_tokens = 1024
    gen_config.temperature = 0.001
    gen_config.top_p = 0.1
    gen_config.repetition_penalty = 1.0

    def extract_terms(user_input):
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
        x = {k:v.to(DEVICE) for k,v in x.items()}

        y = model.generate(**x, generation_config=gen_config)
        y = tokenizer.decode(y[0], skip_special_tokens=False)
        return y

    user_inputs = [
        "角色攻击力+15%，穿透率+15%，无视格挡+12%，+5%暴击率，对所有属性怪物增加16%伤害（可与其他回响叠加）",
        "角色暴击率+10%，暴击伤害+35%，无视格挡+8%，对弱火、无属性怪物增加60%伤害",
        "凤眼蓝使用技能后，立即减少另一个花灵25%正在冷却中的花灵技能剩余冷却时间，且下一个花灵造成的下一次技能伤害提升75%",
    ]
    for user_input in user_inputs:
        print(extract_terms(user_input))
        print('-'*50) 