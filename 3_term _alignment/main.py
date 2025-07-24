#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
os.environ["HTTP_PROXY"] = "http://localhost:1081"
os.environ["HTTPS_PROXY"] = "http://localhost:1081"

from utils import *
import time
from tqdm import tqdm
import json

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, XLMTokenizer
from scipy.spatial.distance import cdist

# Load XLM-Align
# model_name = "microsoft/xlm-align-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model.eval()

from simalign import SentenceAligner

myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

src_sentence = ["#585", "天梯门票", "不足","是否","消耗", "{0}", "购买", "Lv", "2", "门票"]
trg_sentence = ["Ticket", "de", "Escalón", "#585", "insuficiente", "¿", "Deseas", "consumir", "{0}", "para", "comprar", "Nv.", "2", "ticket"]
alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

for matching_method in alignments:
    print(matching_method, ":", alignments[matching_method])

for (i, j) in alignments["itermax"]:
    print(src_sentence[i], trg_sentence[j])
