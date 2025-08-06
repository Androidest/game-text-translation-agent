#%%
from utils import *
from tqdm import tqdm
from p4_RAG.model import TextEmbModel

INPUT_PATH = PATH_DATA / "term_extraction.final.xlsx"
OUTPUT_PATH = PATH_DATA / "term_extraction.final.cleaned.xlsx"
FILTERED_PATH = PATH_DATA / "filtered.xlsx"

src = Sheet(INPUT_PATH)
cleaned = Sheet(OUTPUT_PATH, default_data={ "CN": [], "TERMS": [], "ES": [] }, clear=True)
filtered_sheet = Sheet(FILTERED_PATH, conlumns=["TERMS", "SIMILAR", "DIFF"], clear=True)
text_emb_model = TextEmbModel()

terms_dict = {}
for i, (cn, terms, es) in enumerate(tqdm(src, desc="Cleaning Data:")):
    if terms == "[]":
        continue
    
    if terms in terms_dict:
        sim = text_emb_model.cosine_similarity(terms_dict[terms]["SIMILAR"][0], cn)
        if sim > 0.98:
            terms_dict[terms]["SIMILAR"].append(cn)
            continue
        else:
            terms_dict[terms]["DIFF"].append(cn)
    else:
        terms_dict[terms] = { "SIMILAR": [cn], "DIFF": [] }

    cleaned.append({ "CN": cn, "TERMS": terms, "ES": es })

for terms, cns in terms_dict.items():
    if len(cns["SIMILAR"]) > 1 or len(cns["DIFF"]) > 1:
        filtered_sheet.append({ 
            "TERMS": terms, 
            "SIMILAR": "\n".join(cns["SIMILAR"]),
            "DIFF": "\n".join(cns["DIFF"])
        })

cleaned.save()
filtered_sheet.save()
print(f"Done! Total samples after cleaning: {len(cleaned)}")
print(f"Saved to file {OUTPUT_PATH}")
