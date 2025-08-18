from utils import *
from torch.utils.data import Dataset, DataLoader
from .model import QwenGameTermTokenizer
import torch
import numpy as np
from typing import *

ModelInputDict = Dict[str, torch.Tensor]

class GameTermGenDataset(Dataset):
    def __init__(self, tokenizer:QwenGameTermTokenizer, sheet:Sheet, start:int=0, end:int=-1):
        if end == -1 or end > len(sheet):
            end = len(sheet)

        self.dataframe = sheet[start:end]
        cn = list(self.dataframe["CN"])
        terms = list(self.dataframe["TERMS"])

        samples = tokenizer.create_term_extraction_prompt(cn, terms)

        x = tokenizer(samples, max_length=1024, padding=True, truncation=True)
        print('max_len:', len(x['input_ids'][0]))

        self.input_ids = np.array(x['input_ids'])
        self.attention_mask = np.array(x['attention_mask'])

    def __getitem__(self, index) -> ModelInputDict:
        return {
            "input_ids":torch.from_numpy(self.input_ids[index, :-1]),
            "attention_mask":torch.from_numpy(self.attention_mask[index, :-1]),
            "labels":torch.from_numpy(self.input_ids[index, 1:]),
        }
    
    def __len__(self) -> int:
        return len(self.dataframe)

if __name__ == "__main__":
    MODEL_PATH = get_model_local_path(ModelID.QWEN3)
    DS_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'

    tokenizer = QwenGameTermTokenizer.from_pretrained(MODEL_PATH)
    sheet = Sheet(DS_SHEET_PATH)
    ds = GameTermGenDataset(tokenizer, sheet)

    print(f"ds len: {len(ds)}")
    for batch in DataLoader(ds, batch_size=3, shuffle=True):
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)

        print(batch["input_ids"][0])
        print(batch["labels"][0])
        break