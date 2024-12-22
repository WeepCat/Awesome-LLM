from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional, List, Dict, Union
import torch
from utils.util import apply_chat_template
from .configs import ScoreConfig


class RewardModelScorer:
    def __init__(
        self,
        rm: PreTrainedModel,
        rm_tokenizer: PreTrainedTokenizer,
        config: Optional[ScoreConfig] = None
    ):
        """
        初始化reward model评分器

        Args:
            rm: Reward模型
            rm_tokenizer: Reward模型的tokenizer
        """
        self.rm = rm
        self.rm_tokenizer = rm_tokenizer
        self.config = config or ScoreConfig()
        self.rm.to(self.config.device)
        self.rm.eval()  # 设置为评估模式

    # @lru_cache(maxsize=1024)
    def _process_question(self, question: List[Dict]) -> str:
        """处理问题文本，使用缓存避免重复处理"""
        if self.rm_tokenizer.chat_template is not None:
            return self.rm_tokenizer.apply_chat_template(
                question,
                tokenize=False,
                add_generation_prompt=False
            ).replace(self.rm_tokenizer.bos_token, "")
        return apply_chat_template(question)

    def _batch_tokenize(
            self,
            questions: List[str],
            answers: List[str]
    ) -> dict:
        """批量tokenize输入文本"""
        inputs = self.rm_tokenizer(
            questions,
            answers,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        return {k: v.to(self.config.device) for k, v in inputs.items()}

    @torch.no_grad()
    def get_batched_reward(
            self,
            question: List[Dict],
            answers: List[str],
            return_tensors: bool = True
    ) -> Union[torch.Tensor, List[float]]:
        """
        获取批量回答的reward分数

        Args:
            question: 单个问题
            answers: 回答列表
            return_tensors: 是否返回tensor，False则返回Python列表

        Returns:
            reward分数的tensor或列表
        """
        try:
            questions = [self._process_question(question)] * len(answers)

            all_scores = []
            for i in range(0, len(answers), self.config.batch_size):
                batch_questions = questions[i:i + self.config.batch_size]
                batch_answers = answers[i:i + self.config.batch_size]

                # 获取当前批次的输入
                inputs = self._batch_tokenize(batch_questions, batch_answers)

                outputs = self.rm(**inputs)
                scores = outputs.logits[:, 0]
                all_scores.append(scores)

            # 合并所有批次的结果
            final_scores = torch.cat(all_scores)

            # 返回结果
            if return_tensors:
                return final_scores
            return final_scores.cpu().tolist()

        except Exception as e:
            print(f"Error in get_batched_reward: {str(e)}")
            raise

    def __call__(
            self,
            question: List[Dict],
            answers: List[str],
            return_tensors: bool = True
    ) -> Union[torch.Tensor, List[float]]:
        """使类实例可调用"""
        return self.get_batched_reward(question, answers, return_tensors)
