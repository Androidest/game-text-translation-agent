#%%
import os
os.chdir(os.path.join(os.path.dirname(__file__), "../"))
from utils import *
import re
from tqdm import tqdm

def is_valid_text_cn(text):
    if not text or not isinstance(text, str) or text == "":
        return False
    
    # Define the regex pattern for Unicode ranges of Korean and Japanese:
    # Korean: U+AC00-U+D7AF
    # Japanese Hiragana: U+3040-U+309F
    # Japanese Katakana: U+30A0-U+30FF
    pattern = re.compile(r'[\uAC00-\uD7AF\u3040-\u309F\u30A0-\u30FF]')
    # the text is invalid if it contains Korean or Japanese characters
    return not bool(pattern.search(text))

def is_valid_text_es(text):
    if not text or not isinstance(text, str) or text == "":
        return False
    
    # Define the regex pattern for Unicode ranges of Korean and Japanese:
    # Korean: U+AC00-U+D7AF
    # Japanese Hiragana: U+3040-U+309F
    # Japanese Katakana: U+30A0-U+30FF
    # Chinese: U+4E00-U+9FFF
    pattern = re.compile(r'[\uAC00-\uD7AF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    # the text is invalid if it contains Chinese, Korean or Japanese characters
    return not bool(pattern.search(text))

src = Sheet("data/game_lang_dataset.xlsx")
cleaned = Sheet("data/game_lang_dataset_cleaned.xlsx", default_data={ "CN": [], "ES": [] }, clear=True)
for i, (cn, es) in tqdm(enumerate(src), desc="Cleaning Data:"):
    if is_valid_text_cn(cn) and is_valid_text_es(es):
        cleaned.append({ "CN": cn, "ES": es })
cleaned.save()

