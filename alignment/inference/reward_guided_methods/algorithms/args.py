from typing import List, Dict, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from alignment.inference.reward_guided_methods.reward_scorer import RewardModelScorer
from alignment.inference.reward_guided_methods.configs import GenerationConfig


class ARGS:
    """
    A class for generating text using language models with reward model guidance.
    """

    def __init__(
            self,
            llm: PreTrainedModel,
            llm_tokenizer: PreTrainedTokenizer,
            scorer: RewardModelScorer,
            config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the generator.

        Args:
            llm: Language model for generation
            llm_tokenizer: Tokenizer for the language model
            scorer: Reward model scorer
            config: Generation configuration
        """
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.scorer = scorer
        self.config = config or GenerationConfig()

        # Move model to device
        self.llm.to(self.config.device)

    def _process_prompt(self, prompt: List[Dict]) -> str:
        """Process the input prompt."""
        return self.llm_tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False
        ).replace(self.llm_tokenizer.bos_token, "")

    def _get_generate_kwargs(self, is_first_step: bool) -> dict:
        """Get kwargs for generate method."""
        kwargs = {
            'max_new_tokens': 1,
            'output_scores': True,
            'return_dict_in_generate': True,
            'renormalize_logits': True,
            'return_legacy_cache': True
        }
        if is_first_step:
            kwargs['pad_token_id'] = self.llm_tokenizer.eos_token_id
        return kwargs

    def _get_top_k_tokens(self, scores: torch.Tensor) -> torch.return_types.topk:
        """Get top-k tokens and their probabilities."""
        return torch.topk(scores, self.config.topk)

    def _create_temp_sequences(
            self,
            sequence: torch.Tensor,
            topk_tokens: torch.return_types.topk,
            is_first_step: bool
    ) -> torch.Tensor:
        """Create temporary sequences for reward computation."""
        if is_first_step:
            return topk_tokens.indices.view(-1, 1)
        return torch.cat(
            (sequence.repeat(self.config.topk, 1), topk_tokens.indices.view(-1, 1)),
            dim=1
        )

    def _sample_token(
            self,
            score_tensor: torch.Tensor,
            topk_tokens: torch.return_types.topk
    ) -> torch.Tensor:
        """Sample the next token based on scores."""
        if self.config.mode == 1:
            sampled_id = torch.argmax(score_tensor).item()
        else:  # mode == 2
            sampled_id = torch.distributions.Categorical(logits=score_tensor).sample().item()
        return topk_tokens.indices[sampled_id].view(1, 1)

    @torch.no_grad()
    def generate(self, prompt: List[Dict]) -> dict:
        """
        Generate text using the language model with reward guidance.

        Args:
            prompt: Input prompt for generation

        Returns:
            dict: Contains the generated sequence
        """
        # Process prompt
        llm_prompt = self._process_prompt(prompt)

        # Initialize tensors
        input_ids = self.llm_tokenizer(
            llm_prompt,
            return_tensors='pt'
        ).input_ids.to(self.config.device)
        sequence = torch.empty(
            (1, 0),
            dtype=torch.int64,
            device=self.config.device
        )

        for t in range(self.config.max_sequence_length):
            # Generate next token
            current_input = input_ids if t == 0 else torch.cat(
                (input_ids, sequence),
                dim=1
            )

            # Get generation output
            generate_kwargs = self._get_generate_kwargs(t == 0)
            output = self.llm.generate(inputs=current_input, **generate_kwargs)

            # Get top-k tokens and their probabilities
            topk_tokens = self._get_top_k_tokens(output["scores"][0][0])
            token_probs = topk_tokens.values

            # Create temporary sequences and compute rewards
            temp_sequences = self._create_temp_sequences(sequence, topk_tokens, t == 0)
            rewards = self.scorer(
                prompt,
                self.llm_tokenizer.batch_decode(temp_sequences, skip_special_tokens=True),
                return_tensors=True
            )

            # Compute final scores and sample next token
            score_tensor = token_probs + self.config.w * rewards
            sampled_token = self._sample_token(score_tensor, topk_tokens)

            # Update sequence
            sequence = torch.cat((sequence, sampled_token), dim=1)

            # Check for EOS token
            if sequence[0, -1].item() == self.llm_tokenizer.eos_token_id:
                print(f"EOS BREAK at step {t}")
                break

        return {
            "sequence": self.llm_tokenizer.decode(
                sequence[0],
                skip_special_tokens=True
            )
        }
