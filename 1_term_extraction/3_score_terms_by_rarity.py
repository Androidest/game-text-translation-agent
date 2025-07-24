#%% 
# Imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *
import jieba
import jieba.analyse
import json
from tqdm import tqdm

TERM_EXTRACTOR_DS_PATH = "../data/term_extraction.xlsx"
TERM_DICT_PATH = "../data/term_dict.txt"
TERM_IDF_ITF_PATH = "../data/term_score.xlsx"

#%%
# generate a terminology dictionary as a user dict for the jieba tokenizer 
# from the first term extractor dataset
print("Loading term extractor dataset...")
terms_ds = Sheet(TERM_EXTRACTOR_DS_PATH)
term_set = set()
for _, terms, _ in tqdm(terms_ds, desc="Loading Terms:"):
    for t in json.loads(terms):
        term_set.add(t)

with open(TERM_DICT_PATH, "w", encoding="utf-8") as f:
    for t in term_set:
        f.write(f"{t} 1000\n")

print(f"Term dict saved to file as a user dict for jieba: {TERM_DICT_PATH}")
jieba.load_userdict(TERM_DICT_PATH)
print("jieba loaded the term dict")

#%%
# Calculate the IDF+ITF score for each term: score of rarity
# ITF score: is the inverse frequency of the term in the Chinese part of the term_extraction.xlsx document 
# IDF score: comes from the default IDF dictionary in jieba 
print("Counting tems frequency in the ds document...")
tf_dict = {}
doc_total_words = 0
for cn, _, _ in tqdm(terms_ds, desc="Counting Terms:"):
    # tokenize the text
    words = jieba.lcut(cn)
    doc_total_words += len(words)
    for w in words:
        if w in term_set:
            tf_dict[w] = tf_dict.get(w, 0) + 1

print("Calculating IDF+ITF scores...")
score_dict = {}
idf_dict = jieba.analyse.default_tfidf.idf_freq
max_idf = max(idf_dict.values())
for w, count in tqdm(tf_dict.items(), desc="IDF+ITF:"):
    tf =  count / float(doc_total_words)
    idf = idf_dict.get(w, max_idf + 1/tf) # plus Inverse Term Frequency
    score_dict[w] = idf

print("Sorting terms by IDF+ITF score...")
sorted_terms = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
term_score_sheet = Sheet(TERM_IDF_ITF_PATH, default_data={ "TERM":[], "SCORE":[] }, clear=True)
for w, idf_plus_itf in sorted_terms:
    term_score_sheet.append({"TERM": w, "SCORE": idf_plus_itf})
term_score_sheet.save()
print(f"Terms Scores saved to file: {TERM_IDF_ITF_PATH}")
