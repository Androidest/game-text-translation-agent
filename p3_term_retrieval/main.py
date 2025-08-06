#%% 
from utils import PATH_DATA
from p1_term_extraction import TokenizerBasedTermExtractor
import json

TERM_ALIGNMENT_PATH = PATH_DATA / "term_alignment.merged.final.xlsx"

class TermRetriever(TokenizerBasedTermExtractor):

    def __init__(self, term_alignment_path=TERM_ALIGNMENT_PATH):
        super().__init__(term_alignment_path)
            
    def retrieve(self, text):
        terms = {}
        for word in self.tokenizer.cut(text):
            if word not in terms and word in self.term_set:
                terms[word] = json.loads(self.terms_sheet[word, "TERM_ES"])
        return terms
    
if __name__ == "__main__":
    r = TermRetriever()
    print(r.retrieve("月桂的星级提升到12星\n"))
    print(r.retrieve("拥有1只太古品质卡皮巴拉后可开启"))
    print(r.retrieve("幻兽种族条件不满足"))
