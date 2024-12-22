from transformers import (
    Trainer,
    PreTrainedModel,
)
import torch
import torch.nn as nn


class BradleyTerryRMTrainer(Trainer):
    def compute_loss(self, model: PreTrainedModel, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).logits
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
