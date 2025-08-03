import os

def use_proxy(url:str="http://127.0.0.1:1081"):
    os.environ["http_proxy"] = url
    os.environ["https_proxy"] = url