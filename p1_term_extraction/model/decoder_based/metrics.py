from p1_term_extraction.model.decoder_based.model import QwenGameTermTokenizer
from transformers import EvalPrediction
from typing import *
import torch
import numpy as np
import torch
import jieba
from rouge_chinese import Rouge


class TermMetrics:
    def __init__(self, tokenizer:QwenGameTermTokenizer, metrics:List[str]=["term_set", "rouge"]):
        self.tokenizer = tokenizer
        self.metrics = metrics

    def __call__(self, params:EvalPrediction, compute_result: bool = True)-> Optional[dict[str, float]]:
        pred_ids, true_label_ids = params.predictions, params.label_ids

        pred_ids[pred_ids == self.tokenizer.IGNORE_INDEX] = self.tokenizer.pad_token_id
        true_label_ids[true_label_ids == self.tokenizer.IGNORE_INDEX] = self.tokenizer.pad_token_id

        pred_text = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        true_text = self.tokenizer.batch_decode(true_label_ids, skip_special_tokens=True)

        results = {}
        for metric_name in self.metrics:
            compute_metrics = self.__getattribute__(metric_name)
            results.update(compute_metrics(pred_text, true_text))

        if "term_set" in self.metrics and "rouge" in self.metrics:
            results["rouge-l_recall"] = 0.3 * results["rouge-l"] + 0.7 * results["recall"]

        return results

    def term_set(self, pred_text:List[str], true_text:List[str]) -> Dict[str, float]:
        pred_terms_list = self.tokenizer.get_terms_from_output(pred_text, return_list=True)
        true_terms_list = self.tokenizer.get_terms_from_output(true_text, return_list=True)

        tp = 0
        fp = 0
        fn = 0
        for true_terms, pred_terms in zip(true_terms_list, pred_terms_list):
            if pred_terms is None:
                pred_terms = []

            true_terms = set(true_terms)
            pred_terms = set(pred_terms)

            tp += len(true_terms & pred_terms)
            fp += len(pred_terms - true_terms)
            fn += len(true_terms - pred_terms)
            
            print("true_terms:", true_terms)
            print("pred_terms:", pred_terms)
            print('-'*50)

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "precision":precision * 100,
            "recall":recall * 100,
            "f1":f1 * 100,
        }

        print(",".join([f"{k}:{v:3.3} " for k,v in metrics.items()]))
        return metrics
    
    def rouge(self, pred_text:List[str], true_text:List[str]) -> Dict[str, float]:

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
        for pred, label in zip(pred_text, true_text):
            pred = self.tokenizer.STRUCTURED_OUTPUT_PREFIX + pred
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            print("label:", label)
            print("pred:", pred)
            print('-'*50)

        metrics = {k: float(np.mean(v)) for k, v in score_dict.items()}
        print(",".join([f"{k}:{v:3.3} " for k,v in metrics.items()]))
        return metrics