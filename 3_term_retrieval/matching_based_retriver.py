#%% 
# Imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import jieba
import json
from tqdm import tqdm

TERM_ALIGNMENT_PATH = "../data/term_alignment.merged.final.xlsx"

class TermRetriever:
    def __init__(self, 
            term_alignment_path=TERM_ALIGNMENT_PATH,
        ):
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.FREQ = {}
        self.terms_sheet = Sheet(term_alignment_path)
        for i in tqdm(range(len(self.terms_sheet)), desc="Loading Terms:"):
            self.tokenizer.add_word(
                word=self.terms_sheet[i, "TERM"],
                freq=self.terms_sheet[i, "SCORE"]
            )
        self.terms_sheet.dataframe.set_index("TERM", inplace=True)
            
    def retrieve(self, text):
        terms = {}
        for word in self.tokenizer.cut(text):
            if word in self.terms_sheet.dataframe.index:
                terms[word] = json.loads(self.terms_sheet[word, "TERM_ES"])
        return terms
    
if __name__ == "__main__":
    r = TermRetriever()
    print(r.retrieve("月桂的星级提升到12星\n"))
    print(r.retrieve("拥有1只太古品质卡皮巴拉后可开启"))
    print(r.retrieve("幻兽种族条件不满足"))
