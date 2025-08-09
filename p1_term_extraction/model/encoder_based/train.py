from transformers import TrainingArguments, Trainer, IntervalStrategy, SchedulerType, EvalPrediction
from .model import GameTermBertTokenizer, GameTermBert
from .dataset import GameTermNERDataset
from utils import PATH_DATA, PATH_MODELS, PATH_PROJECT_ROOT, Sheet
from typing import *
import numpy as np
import pdb

LOG_PATH = PATH_PROJECT_ROOT / "p1_term_extraction" / "model" / "logs" / "encoder-based"
MODEL_PATH = PATH_MODELS / "chinese-macbert-base"
SAVE_PATH = PATH_MODELS / "fine-tuned-macbert-game-term-ner"
TRAIN_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'
TEST_SHEET_PATH = PATH_DATA/'term_extraction_test.xlsx'

class MetricsForGameTermBILabels:
    def __init__(self, tokenizer:GameTermBertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, params:EvalPrediction)-> Dict[str, float]:
        pred_logits = params.predictions[0]
        true_label_ids = params.label_ids

        pred_label_ids = np.argmax(pred_logits, axis=-1, keepdims=False)
        true_offset_list = tokenizer.convert_label_ids_to_term_offsets(true_label_ids)
        pred_offset_list = tokenizer.convert_label_ids_to_term_offsets(pred_label_ids)

        tp = 0
        fp = 0
        fn = 0
        for true_offset, pred_offset in zip(true_offset_list, pred_offset_list):
            true_offset = set(true_offset)
            pred_offset = set(pred_offset)

            tp += len(true_offset & pred_offset)
            fp += len(pred_offset - true_offset)
            fn += len(true_offset - pred_offset)

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision":precision,
            "recall":recall,
            "f1":f1,
        }

def train(model:GameTermBert, tokenizer:GameTermBertTokenizer) -> Trainer:
    sheet_train = Sheet(TRAIN_SHEET_PATH)
    ds_train = GameTermNERDataset(tokenizer, sheet_train)
    print(f"ds_train len: {len(ds_train)}, seq_len:{len(ds_train[0]["input_ids"])}")

    sheet_test = Sheet(TEST_SHEET_PATH)
    ds_test = GameTermNERDataset(tokenizer, sheet_test)
    print(f"ds_test len: {len(ds_test)}, seq_len:{len(ds_test[0]["input_ids"])}")

    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        logging_dir=LOG_PATH,
        logging_steps=50,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=300,
        eval_accumulation_steps=1,   # [Important] To prevent gathering all predictions of the entire test set at once and making a single compute_metrics call. 
        save_strategy='best',
        save_total_limit=3,
        metric_for_best_model="f1",   # Return from compute_metrics()
        greater_is_better=True,       # F1 is greater better
        load_best_model_at_end=True,
        
        num_train_epochs=100,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=40,
        optim="adamw_torch",
        weight_decay=5e-3,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.COSINE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        compute_metrics=MetricsForGameTermBILabels(tokenizer), 
    )
    
    trainer.train()
    return trainer

if __name__ == "__main__":
    model:GameTermBert  = GameTermBert.from_pretrained(MODEL_PATH)
    tokenizer:GameTermBertTokenizer = GameTermBertTokenizer.from_pretrained(MODEL_PATH)
    trainer = train(model, tokenizer)

    print("Training Finished!")
    eval_results = trainer.evaluate()
    print(f"======= Best Model Score =======")
    for key, score in eval_results.items():
        if key.startswith("eval_"):
            print(f"{key}: {score:.4f}")

    BEST_MODEL_PATH = SAVE_PATH/'best'
    trainer.model.save_pretrained(BEST_MODEL_PATH)
    trainer.tokenizer.save_pretrained(BEST_MODEL_PATH)
    print(f"Best model saved to: {BEST_MODEL_PATH}")