#%%
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sheet import Sheet
import numpy as np
import os
from tqdm import tqdm
import jieba.posseg as pseg
os.chdir(os.path.dirname(__file__))

def top_tfidf_terms(sheet:Sheet, save_path:str):
    corpus = []
    for (cn, es) in sheet:
        if not isinstance(cn, str):
            continue
        corpus.append(cn)
        
    # 初始化TF-IDF向量化器
    print("Start fitting TFIDF...")
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: pseg.lcut(x),
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 提取TF-IDF值较高的词汇作为候选术语
    print("Calculating mean TFIDF")
    top_terms = []
    for r in tqdm(range(len(corpus))):
        doc = tfidf_matrix[r]
        doc_tfidf = zip(doc.indices, doc.data)
        sorted_tfidf = sorted(doc_tfidf, key=lambda x: x[1], reverse=True)

        for idx, score in sorted_tfidf[:10]:
            w, flag = feature_names[idx]
            if len(w) > 1 and flag[0] == 'n':
                top_terms.append(w)
    top_terms = list(set(top_terms))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for term in top_terms:
            f.write(term + '\n')
    print("Done.")

if __name__ == '__main__':
    sheet = Sheet('TLON语言包ES 20250701.xlsx')
    top_tfidf_terms(sheet, 'terms_dict.txt')