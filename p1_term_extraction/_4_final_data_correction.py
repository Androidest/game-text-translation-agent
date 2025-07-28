#%% 
from utils import *
import jieba
import json
from tqdm import tqdm

TERM_SCORE_FINAL_PATH = PATH_DATA / "term_score.final.xlsx"
TERM_EXTRACTOR_DS_FINAL_PATH = PATH_DATA / "term_extraction.final.xlsx"
TERM_EXTRACTOR_DS_PATH = PATH_DATA / "term_extraction.xlsx"

term_score_final = Sheet(TERM_SCORE_FINAL_PATH)
term_ds_final = Sheet(TERM_EXTRACTOR_DS_FINAL_PATH, default_data={ "CN":[], "TERMS":[], "ES":[] }, clear=True)
terms_ds = Sheet(TERM_EXTRACTOR_DS_PATH)

tokenizer = jieba.Tokenizer()
tokenizer.FREQ = {}
term_set = set()
for term, score in tqdm(term_score_final, desc="Loading Terms:"):
    tokenizer.add_word(term, score)
    term_set.add(term)

for cn, _, es in tqdm(terms_ds, desc="Correcting Extraction:"):
    terms = set([t for t in tokenizer.lcut(cn) if t in term_set])
    terms = json.dumps(list(terms), ensure_ascii=False)
    term_ds_final.append({
        "CN": cn,
        "TERMS": terms,
        "ES": es
    })

term_ds_final.save()
print(f"Correction Done! Saved to file: {TERM_EXTRACTOR_DS_FINAL_PATH}")