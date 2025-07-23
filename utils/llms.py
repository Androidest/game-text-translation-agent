#%%
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

deepseek = ChatOpenAI(
    model='deepseek-chat',
    base_url=os.getenv('MODEL_URL'),
    api_key=os.getenv('MODEL_API_KEY'),
    # streaming=True,  # 保留流式输出功能
    # callbacks=[StreamingStdOutCallbackHandler()]  # 添加流式回调处理器
)