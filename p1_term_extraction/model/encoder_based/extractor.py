from p1_term_extraction.model.encoder_based.model import GameTermBertTokenizer, GameTermBert
from p1_term_extraction.model.data_preprocessing import split_text
from utils import *
from typing import *
import torch

TermList = List[str]

class EncoderBasedTermExtractor:
    def __init__(
            self, 
            model_id:ModelID = ModelID.MACBERT_GAME_TERM, 
            device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):

        self.device = device
        model_path = get_model_local_path(model_id, ModelSrc.MODELSCOPE)
        self.tokenizer = GameTermBertTokenizer.from_pretrained(model_path)
        self.model  = GameTermBert.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract(self, text:str) -> Union[TermList, List[TermList]]:
        x = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        y = self.model(
            input_ids = x["input_ids"].to(self.device), 
            attention_mask = x["attention_mask"].to(self.device)
        )

        label_ids = y.logits.argmax(dim=-1)
        extracted_terms = self.tokenizer.convert_label_ids_to_terms(label_ids, input_text=text, offsets_mapping=x['offset_mapping'])
        
        if isinstance(text, str):
            extracted_terms = extracted_terms[0]

        return extracted_terms


if __name__ == "__main__":
    TEST_SHEET_PATH = PATH_DATA / 'term_extraction_test.xlsx'

    sheet_test = Sheet(TEST_SHEET_PATH)
    extractor = EncoderBasedTermExtractor()

    for i in range(222, 242):
        print("-"*20)
        text = sheet_test[i, "CN"]
        print("Text:", text)
        print("Original terms:", sheet_test[i, "TERMS"])
        print("Extracted terms:", extractor.extract(text))