from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import PATH_MODELS
import torch
from typing import Union, List
import numpy as np

MODEL_PATH = PATH_MODELS / "chinese-macbert-base"

class TextEmbModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading Text Embedding Model: {MODEL_PATH} ...")
        self.max_length = 300
        self.device = device
        self.emb_size = AutoConfig.from_pretrained(MODEL_PATH).hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, force_download=False)
        self.model = AutoModel.from_pretrained(MODEL_PATH).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_text_emb(self, text: Union[str, List[str]]) -> np.ndarray:
        if not text:
            raise ValueError("Input text must be a non-empty string or list of strings.")
        
        # Tokenization
        x = self.tokenizer(
            text, 
            add_special_tokens=True,  # Add [CLS] and [SEP]
            return_tensors="pt", # pytorch tensor
            padding=True, # padding is needed when batching
            truncation=True, 
            max_length=self.max_length
        )
        # Move to device
        inputs = {k: v.to(self.device) for k, v in x.items()}
        # Forward pass
        out = self.model(**inputs)
        # Get CLS token embeddings
        cls_emb = out.last_hidden_state[:,0,:].cpu().numpy()

        return cls_emb
    
    def count_tokens(self, text:str) -> int:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        if not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        x = self.tokenizer.tokenize(text, add_special_tokens=True)
        return len(x)
    
    def cosine_similarity(self, text1:str, text2:str) -> float:
        emb1 = self.get_text_emb(text1).reshape(-1)
        emb2 = self.get_text_emb(text2).reshape(-1)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

if __name__ == "__main__":
    m = TextEmbModel()
    print(m.get_text_emb("统计"))