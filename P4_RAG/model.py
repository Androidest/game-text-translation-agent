from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import PATH_PROJECT_ROOT
import torch
from typing import Union, List
import numpy as np

MODEL_PATH = PATH_PROJECT_ROOT / "P4_RAG" / "chinese-macbert-base"

class TextEmbModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        print("Loading model...")
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

if __name__ == "__main__":
    m = TextEmbModel()
    print(m.count_tokens("统计"))
