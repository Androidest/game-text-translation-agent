#%%
from utils import *
from tqdm import tqdm
from p1_term_extraction import TokenizerBasedTermExtractor
from p4_RAG import TextEmbModel
import json
import re
import numpy as np

TERM_SCORE_FINAL_PATH = PATH_DATA / "term_score.final.xlsx"
INPUT_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"
OUTPUT_PATH = PATH_DATA / "term_extraction_ds.xlsx"
MAX_LEN = 128
MIN_COS_SIM = 0.98
text_emb_model = TextEmbModel()
term_extract_tokenizer = TokenizerBasedTermExtractor(TERM_SCORE_FINAL_PATH)

def extract_terms(text:str):
    extra_terms = re.findall(r'[【\[](.*?)[\]】]', text)
    for term in extra_terms:
        term = re.sub(r'\d', '', term)
        if term != '' and len(term) <= 6 and 'var' not in term:
            term_extract_tokenizer.add(term)
    return term_extract_tokenizer.extract(text)

def split_text(text, max_len:int=128):
    if len(text) <= max_len:
        return [text]
    
    texts = []
    cur_text = ""
    reg = r'([？?！!。.…\)\n\\n”])'
    reg_text = f"{reg}"
    for t in re.split(reg, text):
        new_text = cur_text + t
        if len(new_text) <= max_len or t in reg_text or cur_text == "" or t == '\n':
            cur_text = new_text
        else:
            texts.append(cur_text)
            cur_text = t
    
    if cur_text != "":
        texts.append(cur_text)

    return texts

class SimTextGroups:
    def __init__(self):
        self.groups = []
    
    def add(self, text:str):
        t_emb = text_emb_model.get_text_emb(re.sub(r'\d', '', text)).reshape(-1)
        t_emb = t_emb / np.linalg.norm(t_emb)
        pair = (t_emb, text)

        for g in self.groups:
            cos_sim = np.dot(t_emb, g[0][0])
            if cos_sim >= MIN_COS_SIM:
                g.append(pair)
                return

        self.groups.append([pair])

def process_ds():
    input_sheet = Sheet(INPUT_PATH)
    output_sheet = Sheet(OUTPUT_PATH, conlumns=["CN", "TERMS", "SIMILAR"], clear=True)

    term_dict = {}
    for i in tqdm(range(len(input_sheet)), desc="Processing Data:"):
        cn = input_sheet[i, "CN"]
        for text in split_text(cn, MAX_LEN):
            terms = extract_terms(text)
            terms_json = json.dumps(terms, ensure_ascii=False)
            if terms_json not in term_dict:
                term_dict[terms_json] = SimTextGroups()
            term_dict[terms_json].add(text)

    for terms_json, sim_groups in term_dict.items():    
        for g in sim_groups.groups:
            _, text = g[0]
            sims = "\n".join([t for emb, t in g[1:]]) if len(g) > 1 else ""
            output_sheet.append({ "CN": text, "TERMS": terms_json, "SIMILAR": sims })

    output_sheet.save()
    print(f"Done! Total samples: {len(output_sheet)}")
    print(f"Saved to file {OUTPUT_PATH}")

if __name__ == "__main__":
    process_ds()