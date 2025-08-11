from utils import PATH_DATA, PATH_MODELS, Sheet
from .model import GameTermBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json
from typing import *

TermList = List[str]
ModelInputDict = Dict[str, torch.Tensor]
BIOffsetMap = List[Tuple[int,int]]

class GameTermNERDataset(Dataset):
    def __init__(self, tokenizer:GameTermBertTokenizer, sheet:Sheet, start:int=0, end:int=-1):
        if end == -1 or end > len(sheet):
            end = len(sheet)

        self.dataframe = sheet[start:end]
        cn = list(self.dataframe["CN"])
        labels = list(self.dataframe["BI_LABEL"])

        x = tokenizer(cn, return_offsets_mapping=True, max_length=512, padding=True, truncation=True)
        max_len = len(x['input_ids'][0])
        label_ids = tokenizer.convert_labels_to_ids(labels, x['offset_mapping'], max_len=max_len)

        self.input_ids = np.array(x['input_ids'])
        self.attention_mask = np.array(x['attention_mask'])
        self.label_ids = np.array(label_ids)

    def __getitem__(self, index) -> ModelInputDict:
        return {
            "input_ids":torch.from_numpy(self.input_ids[index]),
            "attention_mask":torch.from_numpy(self.attention_mask[index]),
            "labels":torch.from_numpy(self.label_ids[index]),
        }
    
    def __len__(self) -> int:
        return len(self.dataframe)

if __name__ == "__main__":
    MODEL_PATH = PATH_MODELS/'chinese-macbert-base'
    DS_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'

    tokenizer = GameTermBertTokenizer.from_pretrained(MODEL_PATH)
    sheet = Sheet(DS_SHEET_PATH)
    ds = GameTermNERDataset(tokenizer, sheet)

    print(f"ds len: {len(ds)}")
    for batch in DataLoader(ds, batch_size=3, shuffle=True):
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)
        break