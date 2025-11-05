from utils import *
from torch.utils.data import Dataset, DataLoader
from p1_term_extraction.model.decoder_based.model import QwenGameTermTokenizer, Qwen3ForCausalLM
import json
from typing import *
from transformers.trainer_pt_utils import LabelSmoother
from transformers.data.data_collator import DataCollatorForSeq2Seq

DSOutput = Dict[str, List[int]]

class GameTermGenDataset(Dataset):
    def __init__(self, tokenizer:QwenGameTermTokenizer, sheet:Sheet, start:int=0, end:int=-1, is_generation_eval:bool=False):
        if end == -1 or end > len(sheet):
            end = len(sheet)

        self.is_generation_eval = is_generation_eval
        self.dataframe = sheet[start:end]

        cn = list(self.dataframe["CN"])
        terms = list(self.dataframe["TERMS"])

        if is_generation_eval:
            # For text generation evaluation (Generation Quality)
            requests = tokenizer.apply_term_extraction_template(cn)
            self.inputs = tokenizer(requests, add_special_tokens=False)
            self.labels = tokenizer(terms, add_special_tokens=False).input_ids
        else:
            # For Teacher-Forcing training
            requests, responses = tokenizer.apply_term_extraction_template(cn, terms)
            self.inputs = tokenizer(requests, add_special_tokens=False)
            self.labels = []
            IGNORE_INDEX = LabelSmoother.ignore_index
            targets = tokenizer(responses, add_special_tokens=False)
            for i in range(len(targets.input_ids)):
                self.labels.append([IGNORE_INDEX] * len(self.inputs.input_ids[i]) + targets.input_ids[i])
                self.inputs.input_ids[i].extend(targets.input_ids[i])
                self.inputs.attention_mask[i].extend(targets.attention_mask[i])

    def __getitem__(self, index:int) -> DSOutput:
        return dict(
            input_ids=self.inputs.input_ids[index],
            attention_mask=self.inputs.attention_mask[index],
            labels=self.labels[index],
        )
    
    def __len__(self) -> int:
        return len(self.dataframe)

if __name__ == "__main__":
    MODEL_PATH = get_model_local_path(ModelID.QWEN3)
    TRAIN_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'
    EVAL_SHEET_PATH = PATH_DATA/'term_extraction_test.xlsx'

    tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(MODEL_PATH)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors='pt')

    print("====================== Test GameTermGenDataset =======================")
    sheet = Sheet(TRAIN_SHEET_PATH)
    train_ds = GameTermGenDataset(tokenizer, sheet)
    print(f"ds len: {len(train_ds)}")

    for batch in DataLoader(train_ds, batch_size=3, shuffle=True, collate_fn=data_collator):
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)

        input_text = tokenizer.decode(batch["input_ids"][0])
        print('\n[Input Text]:\n', input_text)
        print()
        print("\n[input_ids]:\n", batch["input_ids"][0])
        print("\n[labels]:\n", batch["labels"][0])
        break

    print("\n\n====================== Test Eval =======================")
    sheet = Sheet(EVAL_SHEET_PATH)
    tokenizer.padding_side = 'left'
    eval_ds = GameTermGenDataset(tokenizer, sheet, is_generation_eval=True)
    model:Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(MODEL_PATH)

    for batch in DataLoader(eval_ds, batch_size=3, shuffle=True, collate_fn=data_collator):
        output_texts = model.generate(**batch)

        input_text = tokenizer.batch_decode(batch["input_ids"])
        print('\n[Input Text For Generation Eval]:\n', input_text[0])

        output_text = tokenizer.batch_decode(output_texts)
        print('\n[Output Text For Generation Eval]:\n', output_text[0])

        batch["labels"][batch["labels"] == -100] = tokenizer.pad_token_id
        batch_terms = tokenizer.get_terms_from_output(tokenizer.batch_decode(batch["labels"]))
        extracted_terms = tokenizer.get_terms_from_output(output_text)
        print('\n[Terms from labels]:\n', batch_terms[0])
        print('\n[Terms from extracted]:\n', extracted_terms[0])
        break