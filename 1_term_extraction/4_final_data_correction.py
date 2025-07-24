#%% 
# Imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import jieba
import json
from tqdm import tqdm

TERM_SCORE_FINAL_PATH = "../data/term_score.final.xlsx"
TERM_EXTRACTOR_DS_FINAL_PATH = "../data/terms_extractor_dataset.final.xlsx"
TERM_EXTRACTOR_DS_PATH = "../data/terms_extractor_dataset.xlsx"

term_score_final = Sheet(TERM_SCORE_FINAL_PATH)
term_ds_final = Sheet(TERM_EXTRACTOR_DS_FINAL_PATH, default_data={ "CN":[], "TERMS":[], "ES":[] }, clear=True)
terms_ds = Sheet(TERM_EXTRACTOR_DS_PATH)

tokenizer = jieba.Tokenizer()
tokenizer.FREQ = {}
term_set = set()
for term, score in tqdm(term_score_final, desc="Loading Terms:"):
    tokenizer.add_word(term, score)
    term_set.add(term)

for i in tqdm(range(len(terms_ds)), desc="Correcting Extraction:"):
    cn = terms_ds[i,"CN"]
    es = terms_ds[i,"ES"]
    terms = set([t for t in tokenizer.lcut(cn) if t in term_set])
    terms = json.dumps(list(terms), ensure_ascii=False)
    term_ds_final[i] = {
        "CN": cn,
        "TERMS": terms,
        "ES": es
    }

term_ds_final.save()
print(f"Correction Done! Saved to file: {TERM_EXTRACTOR_DS_FINAL_PATH}")