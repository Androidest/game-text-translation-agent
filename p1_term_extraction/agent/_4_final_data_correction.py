#%% 
from utils import *
from p1_term_extraction import TokenizerBasedTermExtractor
import json
from tqdm import tqdm

TERM_SCORE_FINAL_PATH = PATH_DATA / "term_score.final.xlsx"
INPUT_PATH = PATH_DATA / "term_extraction.xlsx"
OUTPUT_PATH = PATH_DATA / "term_extraction.final.xlsx"

term_extract_tokenizer = TokenizerBasedTermExtractor(TERM_SCORE_FINAL_PATH)
input_sheet = Sheet(INPUT_PATH)
output_sheet = Sheet(OUTPUT_PATH, conlumns=input_sheet.column_names(), clear=True)

for cn, old_terms, es in tqdm(input_sheet, desc="Correcting Extraction:"):
    corrected_terms = term_extract_tokenizer.extract(cn)
    corrected_terms = json.dumps(list(corrected_terms), ensure_ascii=False)
    output_sheet.append({
        "CN": cn,
        "TERMS": corrected_terms,
        "ES": es
    })

output_sheet.save()
print(f"Correction Done! Saved to file: {OUTPUT_PATH}")