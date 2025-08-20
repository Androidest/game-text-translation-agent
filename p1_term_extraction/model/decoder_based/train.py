from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, IntervalStrategy, SchedulerType, EvalPrediction, TrainerControl, TrainerCallback, TrainerState
from .model import QwenGameTermTokenizer, QwenGameTermLoraModel
from .dataset import GameTermGenDataset
from utils import *
from typing import *
import numpy as np
import pdb
import datetime

BASE_MODEL_ID = ModelID.QWEN3
LORA_MODEL_ID = ModelID.QWEN3_LORA
LOG_PATH = PATH_PROJECT_ROOT / "p1_term_extraction" / "model" / "logs" / "decoder-based"
SAVE_PATH = get_model_local_path(LORA_MODEL_ID, ModelSrc.LOCAL)
TRAIN_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'
TEST_SHEET_PATH = PATH_DATA/'term_extraction_test.xlsx'

class ExtraLogger(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            current_lr = kwargs.get('optimizer').param_groups[0]['lr']
            state.log_history.append({
                "learning_rate": current_lr,
            })

class MetricsForGameTermGen:
    def __init__(self, tokenizer:QwenGameTermTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, params:EvalPrediction)-> Dict[str, float]:
        pdb.set_trace()
        pred_logits = params.predictions[0]
        true_label_ids = params.label_ids

        true_text = self.tokenizer.decode(true_label_ids)
        true_terms_list = self.tokenizer.get_terms_from_output(true_text)

        pred_text = self.tokenizer.decode(pred_logits)
        pred_terms_list = self.tokenizer.get_terms_from_output(pred_text)

        tp = 0
        fp = 0
        fn = 0
        for true_terms, pred_terms in zip(true_terms_list, pred_terms_list):
            true_terms = set(true_terms)
            pred_terms = set(pred_terms)

            tp += len(true_terms & pred_terms)
            fp += len(pred_terms - true_terms)
            fn += len(true_terms - pred_terms)

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision":precision,
            "recall":recall,
            "f1":f1,
        }

def train_decoder() -> Trainer:
    BASE_MODEL_PATH = get_model_local_path(BASE_MODEL_ID)

    tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model  = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    lora_model = QwenGameTermLoraModel(base_model)

    sheet_train = Sheet(TRAIN_SHEET_PATH)
    ds_train = GameTermGenDataset(tokenizer, sheet_train)
    print(f"ds_train len: {len(ds_train)}, seq_len:{len(ds_train[0]["input_ids"])}")

    sheet_test = Sheet(TEST_SHEET_PATH)
    ds_test = GameTermGenDataset(tokenizer, sheet_test, 100, 130)
    print(f"ds_test len: {len(ds_test)}, seq_len:{len(ds_test[0]["input_ids"])}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        logging_dir=LOG_PATH / timestamp,
        logging_steps=10,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=10,
        eval_accumulation_steps=1,   # [Important] To prevent gathering all predictions of the entire test set at once and making a single compute_metrics call. 
        save_strategy='best',
        save_total_limit=3,
        metric_for_best_model="f1",   # Return from compute_metrics()
        greater_is_better=True,       # F1 is greater better
        load_best_model_at_end=True,
        label_names=["labels"],
        
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        weight_decay=1e-4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.COSINE,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        callbacks=[ExtraLogger()],
        compute_metrics=MetricsForGameTermGen(tokenizer), 
    )
    
    trainer.train()

    print("Training Finished!")
    eval_results = trainer.evaluate()
    print(f"======= Best Model Score =======")
    for key, score in eval_results.items():
        if key.startswith("eval_"):
            print(f"{key}: {score:.4f}")

    BEST_MODEL_PATH = str(SAVE_PATH / 'best')
    trainer.model.save_pretrained(BEST_MODEL_PATH)
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    return trainer

if __name__ == "__main__":
    train_decoder()

    