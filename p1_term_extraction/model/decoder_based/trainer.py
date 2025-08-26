from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval", **gen_kwargs):
        self.processing_class.padding_side = 'left'
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        self.processing_class.padding_side = 'right'
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None, **gen_kwargs):
        labels = inputs.pop("labels")
        loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        return loss, generated_tokens, labels 