from p1_term_extraction.model.decoder_based.model import QwenGameTermLoraModel, QwenGameTermTokenizer
from p1_term_extraction.model.data_preprocessing import split_text
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
from typing import *
import torch
import os

TermList = List[str]

class DecoderBasedTermExtractor:
    def __init__(
            self, 
            base_model_id:ModelID = ModelID.QWEN3, 
            lora_model_id:ModelID = ModelID.QWEN3_LORA,
            lora_scr:ModelSrc = ModelSrc.MODELSCOPE,
            device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):

        MODEL_PATH = get_model_local_path(base_model_id)
        MODEL_LORA_PATH = str(get_model_local_path(lora_model_id, lora_scr))
        self.device = device

        self.tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(MODEL_PATH)
        base_model  = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        if lora_scr == ModelSrc.LOCAL and not os.path.exists(MODEL_LORA_PATH):
            # if the source is LOCAL and the model files do not exists
            # create a new lora model and save it locally
            QwenGameTermLoraModel(base_model).save_pretrained(MODEL_LORA_PATH)
        
        self.model:QwenGameTermLoraModel = QwenGameTermLoraModel.from_pretrained(
            model=base_model,
            model_id=str(MODEL_LORA_PATH),
            adapter_name="extractor",
            is_trainable=False,
        )
        self.model.set_adapter("extractor")
        self.model.eval()
        self.model.to(self.device)
        self.model.merge_and_unload()

        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 1024
        gen_config.temperature = 0.001
        gen_config.top_p = 0.1
        gen_config.repetition_penalty = 1.0

    @torch.no_grad()
    def extract(self, text:Union[str, List[str]]) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        text = self.tokenizer.apply_term_extraction_template(text)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            add_special_tokens=False, 
            padding_side='left',
            padding=True,
        )
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        outputs = self.model.generate(**inputs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        terms = self.tokenizer.get_terms_from_output(outputs)

        return terms

if __name__ == "__main__":
    TEST_SHEET_PATH = PATH_DATA / 'term_extraction_test.xlsx'

    sheet_test = Sheet(TEST_SHEET_PATH)
    extractor = DecoderBasedTermExtractor(
        base_model_id=ModelID.QWEN3,
        lora_model_id=ModelID.QWEN3_LORA,
        lora_scr=ModelSrc.LOCAL,
        device='cpu'
    )

    start = 222
    end = 242
    original_terms_list = list(sheet_test[start:end, "TERMS"])
    text_list = list(sheet_test[start:end, "CN"])
    terms_list = extractor.extract(text_list)

    for text, terms, oterms in zip(text_list, terms_list, original_terms_list):
        print("-"*20)
        print("Text:", text)
        print("Original terms:", oterms)
        print("Extracted terms:", terms)