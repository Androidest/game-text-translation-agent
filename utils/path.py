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

class ModelID(Enum):
    LLAMA3_2 = "llama3.2-1b"
    QWEN3 = "qwen3-0.6b"
    QWEN2_5 = "qwen2.5-0.5b"

MODEL_MAPPINGS = {
    ModelID.LLAMA3_2: {
        ModelSrc.HUGGINGFACE: "meta-llama/Llama-3.2-1B-Instruct",
        ModelSrc.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct"
    },
    ModelID.QWEN3: {
        ModelSrc.HUGGINGFACE: "Qwen/Qwen3-0.6B",
        ModelSrc.MODELSCOPE: "Qwen/Qwen3-0.6B"
    },
    ModelID.QWEN2_5: {
        ModelSrc.HUGGINGFACE: "Qwen/Qwen2.5-0.5B-Instruct",
        ModelSrc.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct"
    }
}

DEFAULT_MODEL_SRC = ModelSrc.HUGGINGFACE
if os.getenv('MODEL_SRC'):
    DEFAULT_MODEL_SRC = ModelSrc[os.getenv('MODEL_SRC')]

def get_llm_local_path(
    model_id: ModelID,
    src: ModelSrc = DEFAULT_MODEL_SRC,
):
    routed_id = MODEL_MAPPINGS[model_id][src]

    if src == ModelSrc.HUGGINGFACE:
        final_path = hf_download(routed_id)
    elif src == ModelSrc.MODELSCOPE:
        final_path = ms_download(routed_id)
    return final_path