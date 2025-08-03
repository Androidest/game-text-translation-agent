from utils import Sheet, PATH_DATA
from P4_RAG.model import TextEmbModel
import faiss
import numpy as np
from pathlib import Path
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # An error might occur during faiss index searching

TEXT_SHEET_PATH = PATH_DATA / "rag_text.xlsx"
INDEX_PATH = PATH_DATA / "rag_IVF_COS.index"

class RAG:
    def __init__(
            self, 
            index_path:Path=INDEX_PATH, 
            text_sheet_path:Path=TEXT_SHEET_PATH,
            device:str="cuda" if torch.cuda.is_available() else "cpu"
        ):
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"Loading text sheet from {text_sheet_path}...")
        self.text_sheet = Sheet(text_sheet_path)
        self.model = TextEmbModel(device=device)
        print(f"Model loaded on device:{self.model.device}")

    def search_sim(self, text: str):
        emb = self.model.get_text_emb(text)
        faiss.normalize_L2(emb) # for cosine similarity
        cos_sim, indexes = self.index.search(emb, k=1)
        indexes = np.reshape(indexes, -1)
        cos_sim = np.reshape(cos_sim, -1)
        texts = np.array(self.text_sheet[indexes])
        return texts, cos_sim

if __name__ == "__main__":
    TEST_SHEET_PATH = PATH_DATA / "test.xlsx"
    test_sheet = Sheet(TEST_SHEET_PATH)
    rag = RAG()
    target_text = list(test_sheet[[136,1180,1162,1170,938,1230,1230,1227], "cn"])
    texts, cos_sim = rag.search_sim(target_text)
    # mask = cos_sim >= 0.9
    # texts, cos_sim, target_text = texts[mask], cos_sim[mask], np.array(target_text)[mask]
    for i in range(len(texts)):
        print(f"Target text: {target_text[i]}")
        print(f"CN: {texts[i,0]} -> ES: {texts[i,1]} ")
        print(f"Cosine Similarity: {cos_sim[i]}")
        print("-" * 50)

