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
TERM_TRANSLATION_FINAL_PATH = "../data/term_translation.final.xlsx"
TERM_EXTRACTOR_DS_PATH = "../data/term_extraction.xlsx"

class TermExtractor:
    def __init__(self, 
            term_score_path=TERM_SCORE_FINAL_PATH,
            term_translation_path=TERM_TRANSLATION_FINAL_PATH
        ):
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.FREQ = {}
        self.term_set = set()
        self.term_translation = Sheet(term_translation_path)
        for term, score in tqdm(Sheet(term_score_path), desc="Loading Terms:"):
            self.tokenizer.add_word(term, score)
            self.term_set.add(term)
            
    def extract(self, text):
        terms = [t for t in self.tokenizer.cut(text) if t in self.term_set]
        for t in terms:
            pass