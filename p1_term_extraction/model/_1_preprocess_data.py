#%%
from utils import *
from tqdm import tqdm
from p1_term_extraction import TokenizerBasedTermExtractor
from p4_RAG import TextEmbModel
import json
import re
import numpy as np
import string
from pathlib import Path

TERM_SCORE_FINAL_PATH = PATH_DATA / "term_alignment.merged.final.xlsx"
# TERM_SCORE_FINAL_PATH = PATH_DATA / "term_score.xlsx"
INPUT_TRAIN_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"
OUTPUT_TRAIN_PATH = PATH_DATA / "term_extraction_train.xlsx"
INPUT_TEST_PATH = PATH_DATA / "test.xlsx"
OUTPUT_TEST_PATH = PATH_DATA / "term_extraction_test.xlsx"
MIN_LEN = 4
MAX_LEN = 128
MIN_COS_SIM = 0.97
EN_PUNCTATION = string.punctuation
CN_PUNCTATION = '。，、；：？！…—·ˉ¨‘’“”『』〖〗【】±×÷∶∫∬∭∮∇∆∈∋∝∞∧∨∑∏∪∩∈∉∌⊂⊃⊆⊇≤≥≦≧≡≢≈≒≠≡≤≥＜＞≮≯∷√∛∜∫∬∭∮∇∆∏∑∈∋∝∞∧∨∩∪∅∀∃∄∇∆∏∑∫∬∭∮∴∵∶∷∽∝≅≌≈≒≠≡≤≥＜＞≮≯⊂⊃⊆⊇∈∉∌'
text_emb_model = TextEmbModel()
term_extract_tokenizer = TokenizerBasedTermExtractor(TERM_SCORE_FINAL_PATH)

def extract_terms(text:str):
    extra_terms = re.findall(r'[「【\[](.*?)[\]】」]', text)
    for term in extra_terms:
        # remove any digits
        term = re.sub(r'\d', '', term)
        # do not contain any punctation nor 'var' and length <= 6
        if term != '' and \
            len(term) <= 6 and \
            not bool(re.search(f"[{re.escape(EN_PUNCTATION+CN_PUNCTATION)}]|var", term)):
            # not a stop-word nor an existing term
            if term not in term_extract_tokenizer.tokenizer.FREQ: 
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

def create_BI_label(text:str, terms_json:str):
    label = ["O"] * len(text)

    if terms_json != "[]":
        start = 0
        terms = json.loads(terms_json)
        for term in terms:
            start = text.find(term, start)
            end = start + len(term)
            label[start:end] = ["B"] + ["I"] * (len(term) - 1)
            start = end

    return "".join(label)

class SimTextGroups:
    def __init__(self):
        self.groups = []
    
    def add(self, text:str):
        t_emb = text_emb_model.get_text_emb(re.sub(r'[\d\n(\\n)]', '', text)).reshape(-1)
        t_emb = t_emb / np.linalg.norm(t_emb)
        pair = (t_emb, text)

        for g in self.groups:
            cos_sim = np.dot(t_emb, g[0][0])
            if cos_sim >= MIN_COS_SIM:
                g.append(pair)
                return

        self.groups.append([pair])

def process_ds(input_path:Path, output_path:Path):
    input_sheet = Sheet(input_path)
    output_sheet = Sheet(output_path, conlumns=["CN", "TERMS", "BI_LABEL", "SIMILAR"], clear=True)

    term_dict = {}
    for i in tqdm(range(len(input_sheet)), desc="Processing Data:"):
        cn = input_sheet[i, "CN"]
        for text in split_text(cn, MAX_LEN):
            terms = extract_terms(text)
            if len(terms) == 0 and len(text) < MIN_LEN:
                continue
            terms_json = json.dumps(terms, ensure_ascii=False)
            if terms_json not in term_dict:
                term_dict[terms_json] = SimTextGroups()
            term_dict[terms_json].add(text)

    for terms_json, sim_groups in tqdm(term_dict.items(), desc="Saving Data:"):    
        for g in sim_groups.groups:
            _, text = g[0]
            sims = "\n".join([t for emb, t in g[1:]]) if len(g) > 1 else ""
            label = create_BI_label(text, terms_json)
            output_sheet.append({ "CN": text, "TERMS": terms_json, "BI_LABEL": label, "SIMILAR": sims })

    print(f"Writing to file {output_path}")
    output_sheet.save()
    print(f"Done! Total samples: {len(output_sheet)}")
    print(f"Saved to file {output_path}")

if __name__ == "__main__":
    process_ds(INPUT_TRAIN_PATH, OUTPUT_TRAIN_PATH)
    process_ds(INPUT_TEST_PATH, OUTPUT_TEST_PATH)