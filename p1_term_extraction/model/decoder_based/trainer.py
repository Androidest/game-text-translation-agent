from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval", **gen_kwargs):
        self.processing_class.padding_side = 'left'
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        self.processing_class.padding_side = 'right'
        return results
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None, **gen_kwargs):
        if self.args.predict_with_generate:
            labels = inputs.pop("labels")
            loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
            
            generated_tokens = generated_tokens[:, inputs["input_ids"].size(-1): ]
            
            return loss, generated_tokens.cpu(), labels.cpu()
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)