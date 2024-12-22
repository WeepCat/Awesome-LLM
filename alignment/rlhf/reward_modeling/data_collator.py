from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List


@dataclass
class DefaultRewardDataCollator:
    tokenizer: PreTrainedTokenizer
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            positive = self.tokenizer.apply_chat_template(feature['chosen'], tokenize=False,
                                                          add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
            negative = self.tokenizer.apply_chat_template(feature['rejected'], tokenize=False,
                                                          add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
            positive = self.tokenizer(positive, truncation=True, padding=True)
            negative = self.tokenizer(negative, truncation=True, padding=True)
            merged_features.append(
                {
                    "input_ids": positive["input_ids"],
                    "attention_mask": positive["attention_mask"],
                }
            )
            merged_features.append(
                {
                    "input_ids": negative["input_ids"],
                    "attention_mask": negative["attention_mask"],
                }
            )

        batch = self.tokenizer.pad(
            merged_features,
            return_tensors="pt",
        )

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch
