from pathlib import Path
from enum import Enum
from modelscope.hub.snapshot_download import snapshot_download as ms_download
from huggingface_hub import snapshot_download as hf_download
from dotenv import load_dotenv
import os
load_dotenv()

PATH_PROJECT_ROOT = Path(__file__).parent.parent
PATH_DATA = PATH_PROJECT_ROOT / "data"
PATH_MODELS = PATH_PROJECT_ROOT / "saved_models"

class ModelSrc(Enum):
    HUGGINGFACE = 0
    MODELSCOPE = 1
    LOCAL = 2

class ModelID(Enum):
    MACBERT_BASE = "chinese-macbert-base"
    MACBERT_GAME_TERM = 'fine-tuned-macbert-game-term-ner'
    QWEN3 = "qwen3-0.6b"
    QWEN2_5 = "qwen2.5-0.5b"
    QWEN3_LORA = "qwen3-0.6b-lora"

MODEL_MAPPINGS = {
    ModelID.MACBERT_BASE: {
        ModelSrc.HUGGINGFACE: "hfl/chinese-macbert-base",
        ModelSrc.MODELSCOPE: "androidest/chinese-macbert-base"
    },
    ModelID.MACBERT_GAME_TERM: {
        ModelSrc.MODELSCOPE: "androidest/macbert-game-term-ner",
    },
    ModelID.QWEN3: {
        ModelSrc.HUGGINGFACE: "Qwen/Qwen3-0.6B",
        ModelSrc.MODELSCOPE: "Qwen/Qwen3-0.6B"
    },
    ModelID.QWEN2_5: {
        ModelSrc.HUGGINGFACE: "Qwen/Qwen2.5-0.5B-Instruct",
        ModelSrc.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct"
    },
    ModelID.QWEN3_LORA: {
        ModelSrc.MODELSCOPE: "androidest/Qwen3-0.6B-game-term-lora",
    },
}

DEFAULT_MODEL_SRC = ModelSrc.HUGGINGFACE
if os.getenv('MODEL_SRC'):
    DEFAULT_MODEL_SRC = ModelSrc[os.getenv('MODEL_SRC')]

def get_model_local_path(
    model_id: ModelID,
    src: ModelSrc = DEFAULT_MODEL_SRC,
):
    if model_id not in MODEL_MAPPINGS:
        raise KeyError(f"Invalid model ID '{model_id}'.")
    if src not in MODEL_MAPPINGS[model_id] and src != ModelSrc.LOCAL:
        raise KeyError(f"Invalid model source '{src}' for model ID '{model_id}'.")

    if src == ModelSrc.HUGGINGFACE:
        routed_id = MODEL_MAPPINGS[model_id][src]
        final_path = hf_download(routed_id)
    elif src == ModelSrc.MODELSCOPE:
        routed_id = MODEL_MAPPINGS[model_id][src]
        final_path = ms_download(routed_id)
    elif src == ModelSrc.LOCAL:
        final_path = PATH_MODELS / model_id.value
        
    return final_path