#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import *
from tqdm import tqdm

src = Sheet("../data/term_extraction.final.xlsx")
cleaned = Sheet("../data/term_extraction.final.cleaned.xlsx", default_data={ "CN": [], "TERMS": [], "ES": [] }, clear=True)
for i, (cn, terms, es) in tqdm(enumerate(src), desc="Cleaning Data:"):
    if terms == "[]":
        continue
    cleaned.append({ "CN": cn, "TERMS": terms, "ES": es })
cleaned.save()
print(f"Done! Total cleaned data: {len(cleaned)}")
print(f"Saved to file {"../data/term_extraction.final.cleaned.xlsx"}")