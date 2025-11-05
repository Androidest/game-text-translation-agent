#%%
from utils import *
import re
from tqdm import tqdm

INPUT_SHEET_PATH = PATH_DATA / "game_lang_dataset.xlsx"
OUTPUT_SHEET_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"

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

if __name__ == "__main__":
    src = Sheet(INPUT_SHEET_PATH)
    cleaned = Sheet(OUTPUT_SHEET_PATH, default_data={ "CN": [], "ES": [] }, clear=True)
    for cn, es in tqdm(src, desc="Cleaning Data"):
        if is_valid_text_cn(cn) and is_valid_text_es(es):
            cleaned.append({ "CN": cn, "ES": es })
    cleaned.save() 

