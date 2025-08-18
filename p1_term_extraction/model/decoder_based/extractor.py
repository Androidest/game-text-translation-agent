from p1_term_extraction.model.decoder_based.model import QwenGameTermLora
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

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        base_model  = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        if lora_scr == ModelSrc.LOCAL and not os.path.exists(MODEL_LORA_PATH):
            # if the source is LOCAL and the model files do not exists
            # create a new lora model and save it locally
            QwenGameTermLora(base_model).save_pretrained(MODEL_LORA_PATH)
        
        self.model:QwenGameTermLora = QwenGameTermLora.from_pretrained(
            model=base_model,
            model_id=str(MODEL_LORA_PATH),
            adapter_name="extractor",
            is_trainable=False,
        )
        self.model.set_adapter("extractor")
        self.model.eval()
        self.model.to(self.device)
        self.model.merge_and_unload()

        # TODO 要挪到train.py
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 1024
        gen_config.temperature = 0.001
        gen_config.top_p = 0.1
        gen_config.repetition_penalty = 1.0

    @torch.no_grad()
    def extract(self, text):
        # text = tokenizer.apply_chat_template(
        #     conversation = [
        #         {"role": "user", "content": user_input}
        #     ],
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     enable_thinking=False
        # )
        
        text = self.model.create_term_extraction_prompt(text)
        
        x = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        x = {k:v.to(self.device) for k,v in x.items()}

        y = self.model.generate(**x)
        y = self.tokenizer.decode(y[0], skip_special_tokens=False)

        terms = self.model.get_terms_from_output(y)
        return terms

if __name__ == "__main__":
    TEST_SHEET_PATH = PATH_DATA / 'term_extraction_test.xlsx'

    sheet_test = Sheet(TEST_SHEET_PATH)
    extractor = DecoderBasedTermExtractor(
        base_model_id=ModelID.QWEN3,
        lora_model_id=ModelID.QWEN3_LORA,
        lora_scr=ModelSrc.LOCAL
    )

    for i in range(222, 242):
        print("-"*20)
        text = sheet_test[i, "CN"]
        print("Text:", text)
        print("Original terms:", sheet_test[i, "TERMS"])
        print("Extracted terms:", extractor.extract(text))