from p4_RAG.model import TextEmbModel
from utils import PATH_DATA, Sheet
import numpy as np
from tqdm import tqdm
import faiss

INPUT_SHEET_PATH = PATH_DATA / "rag_text.xlsx"
INDEX_PATH = PATH_DATA / "rag_IVF_COS.index"
batch_size = 64

print("Loading sheets...")
model = TextEmbModel()
input_sheet = Sheet(INPUT_SHEET_PATH)
index = faiss.IndexIVFFlat(
    faiss.IndexFlatIP(model.emb_size), # basic index quantizer (IndexFlatIP:Inner Product, IndexFlatL2:L2 Dist)
    model.emb_size, 
    500,
    faiss.METRIC_INNER_PRODUCT, # with this metric, index.search will return inner product, not L2 dist
    )

vectors = np.zeros((len(input_sheet), model.emb_size), dtype=np.float32)
for i in tqdm(range(0, len(input_sheet), batch_size), desc="Get Embeddings:"):
    end = min(i+batch_size, len(input_sheet))
    text = list(input_sheet[i:end-1, "CN"])
    emb = model.get_text_emb(text)
    vectors[i:end] = emb

print("Training...")
faiss.normalize_L2(vectors) # First L2 Norm then Inner Product = Cosine Similarity
index.train(vectors)
print("Indexing...")
index.add(vectors)
faiss.write_index(index, str(INDEX_PATH))
print("Done!")

