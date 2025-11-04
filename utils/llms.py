#%%
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def init_llms():
    llm_dict = {}
    i = 0
    while True:
        name = os.getenv(f'MODEL_NAME_{i}')
        model_id = os.getenv(f'MODEL_ID_{i}')
        base_url=os.getenv(f'MODEL_URL_{i}')
        api_key=os.getenv(f'MODEL_API_KEY_{i}')
        if not all([name, model_id, base_url, api_key]):
            break

        llm_dict[name] = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            # streaming=True,  # 保留流式输出功能
            # callbacks=[StreamingStdOutCallbackHandler()]  # 添加流式回调处理器
        )
        i += 1
    
    return llm_dict

def get_llm(name:str) -> ChatOpenAI:
    if name not in _llms:
        return None
    
    return _llms[name]

_llms = init_llms()
if len(_llms) > 0:
    llm_names = list(_llms.keys())
    default_llm = list(_llms.values())[0]
