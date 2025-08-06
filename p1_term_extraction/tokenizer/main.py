#%% 
from utils import Sheet, PATH_DATA
import jieba
from tqdm import tqdm
import re

class TokenizerBasedTermExtractor:
    def __init__(self, term_score_sheet_path:str):
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.FREQ = {}
        self.terms_sheet = Sheet(term_score_sheet_path)
        self.term_set = set()
        for i in tqdm(range(len(self.terms_sheet)), desc="Loading Terms:"):
            term = self.terms_sheet[i, "TERM"]
            score = self.terms_sheet[i, "SCORE"]
            term = re.sub(r'\d', "", term)
            self.add(term, score)

    def add(self, term:str, score:float=14.0):
        if term in self.tokenizer.FREQ or term == "":
            return
        
        self.term_set.add(term)
        self.tokenizer.add_word(
            word=term,
            freq=score
        )

    def extract(self, text):
        terms = []
        for word in self.tokenizer.cut(text):
            if word in self.term_set:
                terms.append(word)
        return terms
    
if __name__ == "__main__":
    r = TokenizerBasedTermExtractor(PATH_DATA / "term_score.final.xlsx")
    print(r.extract("月桂的星级提升到12星\\n"))
    print(r.extract("拥有1只太古品质卡皮巴拉后可开启"))
    print(r.extract("幻兽种族条件不满足"))
    print(r.extract("<color=gold1>S13</color>级的史诗品质装备\\n随机获得(武器，头盔，手套）3个部位之一的升级材料"))
