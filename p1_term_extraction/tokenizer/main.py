#%% 
from utils import Sheet, PATH_DATA
import jieba
from tqdm import tqdm

class TokenizerBasedTermExtractor:
    def __init__(self, term_score_sheet_path:str):
        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.FREQ = {}
        self.terms_sheet = Sheet(term_score_sheet_path)
        for i in tqdm(range(len(self.terms_sheet)), desc="Loading Terms:"):
            self.tokenizer.add_word(
                word=self.terms_sheet[i, "TERM"],
                freq=self.terms_sheet[i, "SCORE"]
            )
        self.terms_sheet.dataframe.set_index("TERM", inplace=True)
            
    def extract(self, text):
        terms = []
        for word in self.tokenizer.cut(text):
            if word in self.terms_sheet.dataframe.index:
                terms.append(word)
        return terms
    
if __name__ == "__main__":
    r = TokenizerBasedTermExtractor(PATH_DATA / "term_score.final.xlsx")
    print(r.extract("月桂的星级提升到12星\n"))
    print(r.extract("拥有1只太古品质卡皮巴拉后可开启"))
    print(r.extract("幻兽种族条件不满足"))
