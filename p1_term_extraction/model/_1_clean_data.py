#%%
from utils import *
from tqdm import tqdm
from p4_RAG import TextEmbModel

INPUT_PATH = PATH_DATA / "game_lang_dataset_cleaned.xlsx"
TERM_SCORE_FINAL_PATH = PATH_DATA / "term_score.final.xlsx"
OUTPUT_PATH = PATH_DATA / "term_extraction_ds.xlsx"

src_sheet = Sheet(INPUT_PATH)
term_sheet = Sheet(TERM_SCORE_FINAL_PATH)
output_sheet = Sheet(OUTPUT_PATH, default_data={ "CN": [], "TERMS": [], "SIMILAR": [] }, clear=True)
text_emb_model = TextEmbModel()

terms_dict = {}
for i, (cn, terms, es) in enumerate(tqdm(src_sheet, desc="Cleaning Data:")):
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

    output_sheet.append({ "CN": cn, "TERMS": terms, "ES": es })

for terms, cns in terms_dict.items():
    if len(cns["SIMILAR"]) > 1 or len(cns["DIFF"]) > 1:
        filtered_sheet.append({ 
            "TERMS": terms, 
            "SIMILAR": "\n".join(cns["SIMILAR"]),
            "DIFF": "\n".join(cns["DIFF"])
        })

output_sheet.save()
filtered_sheet.save()
print(f"Done! Total samples after cleaning: {len(output_sheet)}")
print(f"Saved to file {OUTPUT_PATH}")
