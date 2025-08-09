from transformers import TrainingArguments, Trainer, IntervalStrategy, SchedulerType
from .model import GameTermBertTokenizer, GameTermBert
from .dataset import GameTermNERDataset
from utils import PATH_DATA, PATH_MODELS, PATH_PROJECT_ROOT, Sheet
from typing import *
import torch

LOG_PATH = PATH_PROJECT_ROOT / "p1_term_extraction" / "model" / "logs" / "encoder-based"
MODEL_PATH = PATH_MODELS / "chinese-macbert-base"
SAVE_PATH = PATH_MODELS / "fine-tuned-macbert-game-term-ner"
TRAIN_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'
TEST_SHEET_PATH = PATH_DATA/'term_extraction_test.xlsx'

def compute_metrics(
    pred_logits:torch.Tensor, 
    true_label_ids:torch.Tensor, 
    tokenizer:GameTermBertTokenizer,
)-> Dict[str, float]:
    
    true_label_ids = true_label_ids.cpu().numpy()
    pred_label_ids = pred_logits.cpu().numpy().argmax(dim=-1, keepdim=False)
    true_offset = set(tokenizer.convert_label_ids_to_term_offsets(true_label_ids))
    pred_offset = set(tokenizer.convert_label_ids_to_term_offsets(pred_label_ids))

    tp = len(true_offset & pred_offset)
    fp = len(pred_offset - true_offset)
    fn = len(true_offset - pred_offset)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision":precision,
        "recall":recall,
        "f1":f1,
    }

def train(model:GameTermBert, tokenizer:GameTermBertTokenizer) -> Trainer:
    sheet_train = Sheet(TRAIN_SHEET_PATH)
    ds_train = GameTermNERDataset(tokenizer, sheet_train)
    print(f"ds_train len: {len(ds_train)}")

    sheet_test = Sheet(TEST_SHEET_PATH)
    ds_test = GameTermNERDataset(tokenizer, sheet_test)
    print(f"ds_test len: {len(ds_test)}")

    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        logging_dir=LOG_PATH,
        logging_steps=20,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        save_strategy='best',
        save_total_limit=3,
        metric_for_best_model="f1",   # Return from compute_metrics()
        greater_is_better=True,       # F1 is greater better
        load_best_model_at_end=True,
        
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        optim="adamw_torch",
        weight_decay=1e-3,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.COSINE,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        compute_metrics=lambda p: compute_metrics(p[0], p[1], tokenizer), 
    )
    
    trainer.train()
    return trainer

if __name__ == "__main__":
    model:GameTermBert  = GameTermBert.from_pretrained(MODEL_PATH)
    tokenizer:GameTermBertTokenizer = GameTermBertTokenizer.from_pretrained(MODEL_PATH)
    trainer = train(model, tokenizer)