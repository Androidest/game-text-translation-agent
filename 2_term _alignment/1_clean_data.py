#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import *
from tqdm import tqdm

INPUT_PATH = "../data/term_extraction.final.xlsx"
OUTPUT_PATH = "../data/term_extraction.final.cleaned.xlsx"

src = Sheet(INPUT_PATH)
cleaned = Sheet(OUTPUT_PATH, default_data={ "CN": [], "TERMS": [], "ES": [] }, clear=True)

term_groups = set()
for i, (cn, terms, es) in tqdm(enumerate(src), desc="Cleaning Data:"):
    if terms == "[]":
        continue
    if terms in term_groups:
        continue

    term_groups.add(terms)
    cleaned.append({ "CN": cn, "TERMS": terms, "ES": es })

cleaned.save()
print(f"Done! Total samples after cleaning: {len(cleaned)}")
print(f"Saved to file {OUTPUT_PATH}")