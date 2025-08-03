import os

def use_proxy(url:str="http://127.0.0.1:1081"):
    os.environ["http_proxy"] = url
    os.environ["https_proxy"] = url

def capitalize_first_char(s:str) -> str:
    if not s:
        return s
    first_char = s[0]
    if first_char.isalpha():
        return first_char.upper() + s[1:]
    else:
        return s
