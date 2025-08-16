#%% 
from utils import PATH_DATA
from p1_term_extraction import TokenizerBasedTermExtractor
import json
import pdb

TERM_ALIGNMENT_PATH = PATH_DATA / "term_alignment.merged.final.xlsx"

class TermRetriever(TokenizerBasedTermExtractor):

    def __init__(self, term_alignment_path=TERM_ALIGNMENT_PATH):
        super().__init__(term_alignment_path)
            
    def retrieve(self, text):
        terms = {}
        for word in self.tokenizer.cut(text):
            if word not in terms \
                and word in self.term_set \
                and word in self.terms_sheet.dataframe.index:
                trans = self.terms_sheet[word, "TERM_ES"]
                if not isinstance(trans, str):
                    raise ValueError(f"Duplicate Term Translation for [{word}]:\n{self.terms_sheet[word, "TERM_ES"]}")
                if trans == "":
                    continue
                terms[word] = json.loads(self.terms_sheet[word, "TERM_ES"])
        return terms
    
if __name__ == "__main__":
    r = TermRetriever()
    print(r.retrieve("月桂的星级提升到12星\\n"))
    print(r.retrieve("拥有1只太古品质卡皮巴拉后可开启"))
    print(r.retrieve("幻兽种族条件不满足"))
    print(r.retrieve("<color=gold1>S13</color>级的史诗品质装备\\n随机获得(武器，头盔，手套）3个部位之一的升级材料"))
    print(r.retrieve("暴击-10%"))
    print(r.retrieve("<color=gold1>更新后，所有冒险者花灵编队的战力都会有不同程度的提升。</color>"))
