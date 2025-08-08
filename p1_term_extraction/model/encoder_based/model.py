from transformers import BertTokenizerFast, BertConfig, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from utils import PATH_MODELS, PATH_DATA, Sheet
import torch
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

MODEL_PATH = PATH_MODELS / "chinese-macbert-base"
SAVE_PATH = PATH_MODELS / "fine-tuned-macbert-game-term-ner"
LABEL_DICT:Dict[str, int] = {
    "B":0,
    "I":1,
    "O":2,
}

class GameTermBertTokenizer(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_labels_to_ids(
        self, 
        labels:Union[str, List[str]], 
        offsets_mapping:Union[List[Tuple[int,int]], List[List[Tuple[int,int]]]], 
        return_tensors:bool=False,
    ) -> Union[torch.Tensor, List[List[int]]]:
        
        if isinstance(labels, str):
            labels = [labels]
        
        results = []
        for l, o in zip(labels, offsets_mapping):
            results.append(self._tokenize_labels(l, o))

        results = np.array(results)
        if return_tensors:
            return torch.tensor(results, dtype=torch.int)

        return results
    
    def _tokenize_labels(
        self, 
        labels:str, 
        offsets_mapping:List[Tuple[int,int]], 
    ) -> List[List[int]]:
        
        ids = []
        for start, end in offsets_mapping:
            if start == 0 and end == 0:
                ids.append(-100)
                continue

            label_name = labels[start]

            if label_name not in LABEL_DICT:
                raise ValueError("The given labels are not in BI format, must be like: 'OOOBIIOBOO'")

            label_id = LABEL_DICT[label_name]
            ids.append(label_id)
        
        return ids

    def convert_label_ids_to_terms(
        self,
        label_ids:Union[torch.Tensor, List[List[int]]], 
        input_text:Union[str, List[str]], 
        offsets_mapping:Union[List[Tuple[int,int]], List[List[Tuple[int,int]]]], 
    ):
        if isinstance(input_text, str):
            input_text = [input_text]

        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.cpu().numpy()
        
        results = []
        for l, t, o in zip(label_ids, input_text, offsets_mapping):
            results.append(self._extract_terms(l, t, o))

        return results
    
    def _extract_terms(
        self, 
        label_ids:List[int],
        input_text:str, 
        offsets_mapping:List[Tuple[int,int]], 
    ) -> List[List[int]]:
        
        terms = []
        cur_term = ""
        for label_id, (start, end) in zip(label_ids, offsets_mapping):
            if start == 0 and end == 0:
                continue

            if label_id == LABEL_DICT['B']:
                if cur_term != "":
                    terms.append(cur_term)
                cur_term = input_text[start:end]
            elif label_id == LABEL_DICT['I']:
                if cur_term == "":
                    raise ValueError(f"Invalid label I in label_ids: {label_ids}")
                    #TODO check valid label id or ignore 
                cur_term += input_text[start:end]
            elif label_id == LABEL_DICT['O']:
                if cur_term != "":
                    terms.append(cur_term)
                    cur_term = ""
        
        if cur_term != "":
            terms.append(cur_term)

        return terms

class GameTermBert(BertPreTrainedModel):
    def __init__(self, config:BertConfig):
        super().__init__(config=config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, len(LABEL_DICT))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kargs,
    ) -> SequenceClassifierOutput:
        
        kargs["return_dict"] = True
        kargs["output_hidden_states"] = True

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kargs,
        )

        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)

        loss = None
        if labels:
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, len(LABEL_DICT)), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    sheet = Sheet(PATH_DATA/'term_extraction_train.xlsx')
    model:GameTermBert  = GameTermBert.from_pretrained(MODEL_PATH)
    tokenizer:GameTermBertTokenizer = GameTermBertTokenizer.from_pretrained(MODEL_PATH)

    text, terms, labels, _ =  sheet[54373-2] #73259
    print(f"Test text: {text}")
    print(f"Test terms: {terms}")
    print(f"Test labels: {labels}")
    print()

    x = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    label_ids = tokenizer.convert_labels_to_ids(labels, offsets_mapping=x['offset_mapping'], return_tensors=True)
    print("label_ids:", label_ids.numpy())
    extracted_terms = tokenizer.convert_label_ids_to_terms(label_ids, input_text=text, offsets_mapping=x['offset_mapping'])
    print("extracted_terms:", extracted_terms)

    y = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
