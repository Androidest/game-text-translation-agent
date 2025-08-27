from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    IntervalStrategy,
    SchedulerType,
    TrainerControl,
    TrainerCallback,
    TrainerState
)

from transformers.data.data_collator import DataCollatorForSeq2Seq
from .model import QwenGameTermTokenizer, QwenGameTermLoraModel
from .dataset import GameTermGenDataset
from .trainer import CustomTrainer
from .metrics import TermMetrics
from utils import *
from typing import *
import datetime

BASE_MODEL_ID = ModelID.QWEN3
LORA_MODEL_ID = ModelID.QWEN3_LORA
LOG_PATH = PATH_PROJECT_ROOT / "p1_term_extraction" / "model" / "logs" / "decoder-based"
SAVE_PATH = get_model_local_path(LORA_MODEL_ID, ModelSrc.LOCAL)
TRAIN_SHEET_PATH = PATH_DATA/'term_extraction_train.xlsx'
TEST_SHEET_PATH = PATH_DATA/'term_extraction_test.xlsx'
IGNORE_INDEX = -100

class ExtraLogger(TrainerCallback):
    def on_step_end(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            current_lr = kwargs.get('optimizer').param_groups[0]['lr']
            state.log_history.append({
                "learning_rate": current_lr,
            })

def train_decoder() -> Seq2SeqTrainer:
    BASE_MODEL_PATH = get_model_local_path(BASE_MODEL_ID)

    tokenizer:QwenGameTermTokenizer = QwenGameTermTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model  = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    lora_model = QwenGameTermLoraModel(base_model)

    gen_config:GenerationConfig = base_model.generation_config
    gen_config.max_new_tokens = 128
    gen_config.temperature = 0.1
    gen_config.top_p = 0.1
    gen_config.repetition_penalty = 1.0

    sheet_train = Sheet(TRAIN_SHEET_PATH)
    ds_train = GameTermGenDataset(tokenizer, sheet_train)
    print(f"ds_train len: {len(ds_train)}")

    sheet_test = Sheet(TEST_SHEET_PATH)
    ds_test = GameTermGenDataset(tokenizer, sheet_test, is_generation_eval=True)
    print(f"ds_test len: {len(ds_test)}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_PATH,
        overwrite_output_dir=True,
        logging_dir=LOG_PATH / timestamp,
        logging_steps=5,

        predict_with_generate=True,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=50,
        eval_accumulation_steps=1,   # [Important] To prevent gathering all predictions of the entire test set at once and making a single compute_metrics call. 
        save_strategy='best',
        save_total_limit=3,
        metric_for_best_model="rouge-l_recall",   # Return from compute_metrics()
        # metric_for_best_model="f1",   # Return from compute_metrics()
        greater_is_better=True,       # F1 is greater better
        load_best_model_at_end=True,
        label_names=["labels"],
        
        num_train_epochs=1,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        optim="adamw_torch",
        weight_decay=1e-4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.COSINE,
    )
    
    trainer = CustomTrainer(
        model=lora_model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors='pt'),
        compute_metrics=TermMetrics(tokenizer, metrics=["term_set", "rouge"]),
        callbacks=[ExtraLogger()],
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
    gen_config.save_pretrained(BEST_MODEL_PATH)
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    return trainer

if __name__ == "__main__":
    train_decoder()

    