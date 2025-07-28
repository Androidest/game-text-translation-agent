#%%
from utils import *
from tqdm import tqdm
import json
import re

INPUT_PATH = PATH_DATA / "term_alignment.xlsx"
SCORE_PATH = PATH_DATA / "term_score.final.xlsx"
OUTPUT_PATH = PATH_DATA / "term_alignment.merged.xlsx"

def spanish_plural_to_singular(text):
    # 规则1: 单词以元音+s结尾(如casas -> casa)
    text = re.sub(r'([aeiouáéíóú])s\b', r'\1', text, flags=re.IGNORECASE)
    
    # 规则2: 单词以辅音+es结尾(如flores -> flor)
    text = re.sub(r'([^aeiouáéíóú])es\b', r'\1', text, flags=re.IGNORECASE)
    
    # 规则3: 单词以ces结尾(可能是z的复数形式，如luces -> luz)
    text = re.sub(r'ces\b', 'z', text, flags=re.IGNORECASE)
    
    # 规则4: 处理一些特殊结尾
    text = re.sub(r'ques\b', 'que', text, flags=re.IGNORECASE)  # 如tanques -> tanque
    text = re.sub(r'gues\b', 'gue', text, flags=re.IGNORECASE)  # 如jugues -> jugue
    
    # 规则5: 处理以"a"或"o"结尾的单词，可能涉及重音变化
    text = re.sub(r'([áéíóú])s\b', lambda m: m.group(1).lower() + '', text)
    
    return text

def is_valid_or_new(term_es_arr, es):
    if es == "":
        return False
    
    es = spanish_plural_to_singular(es.lower())
    for t in term_es_arr:
        t = spanish_plural_to_singular(t.lower())
        if es in t:
            return False
    
    return True

if __name__ == "__main__":
    src = Sheet(INPUT_PATH)
    term_score = Sheet(SCORE_PATH)
    term_score.dataframe.set_index("TERM", inplace=True)
    merged = Sheet(OUTPUT_PATH, default_data={ "TERM": [], "TERM_ES": [], "SCORE": [] }, clear=True)

    align_dict = {}
    for i in tqdm(range(len(src)), desc="Merging Data:"):
        terms_es = json.loads(src[i, "TERMS_ES"])
        for term, es in terms_es.items():
            if term not in align_dict:
                score = term_score[term]["SCORE"]
                align_dict[term] = { "TERM": term, "TERM_ES": [], "SCORE": score }
            if is_valid_or_new(align_dict[term]["TERM_ES"], es):
                align_dict[term]["TERM_ES"].append(es)

    sorted_terms = sorted(align_dict.values(), key=lambda x:len(x["TERM_ES"]))
    for i in tqdm(range(len(sorted_terms)), desc="Writing to file:"):
        line = sorted_terms[i]
        indent = 4 if len(line["TERM_ES"]) > 1 else None
        line["TERM_ES"] = json.dumps(line["TERM_ES"], indent=indent, ensure_ascii=False)
        merged[i] = line

    merged.save()
    print(f"Saved to file {OUTPUT_PATH}")
