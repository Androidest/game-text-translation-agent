from utils import *
from torch.utils.data import Dataset, DataLoader
from .model import QwenGameTermTokenizer
import torch
import numpy as np
from typing import *
from transformers.trainer_pt_utils import LabelSmoother

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
        print('max_len:', len(x.input_ids[0]))

        self.input_ids = np.array(x.input_ids)
        self.attention_mask = np.array(x.attention_mask)

        IM_START_ID = tokenizer('<|im_start|>').input_ids[0]
        IM_END_ID = tokenizer('<|im_end|>').input_ids[0]
        NL_ID = tokenizer('\n').input_ids[0]
        IGNORE_ID = LabelSmoother.ignore_index
        
        labels = np.full_like(self.input_ids, fill_value=IGNORE_ID)
        labels[self.input_ids == IM_START_ID] = IM_START_ID
        labels[self.input_ids == IM_END_ID] = IM_END_ID
        self.labels = labels

        for i in range(labels.shape[0]):
            end_count = 0
            start_count = 0
            for j in reversed(range(labels.shape[1])):
                id = self.input_ids[i, j]
                if id == IM_END_ID:
                    end_count += 1
                elif id == IM_START_ID:
                    start_count += 1
                elif end_count == 1 and start_count == 0:
                    labels[i, j] = self.input_ids[i, j]
                elif end_count == start_count and id == NL_ID:
                    labels[i, j] = NL_ID

    def __getitem__(self, index) -> ModelInputDict:
        return {
            "input_ids":torch.from_numpy(self.input_ids[index]),
            "attention_mask":torch.from_numpy(self.attention_mask[index]),
            "labels":torch.from_numpy(self.labels[index]),
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