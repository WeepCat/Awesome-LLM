from dataclasses import dataclass


@dataclass
class ScoreConfig:
    """Generation configuration parameters"""
    max_length: int = 4096
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class GenerationConfig:
    """Generation configuration parameters"""
    topk: int = 10
    max_sequence_length: int = 10
    mode: int = 1  # 1: argmax, 2: categorical
    w: float = 1.0
    device: str = "cpu"
