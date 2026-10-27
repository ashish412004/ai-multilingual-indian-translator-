"""
Evaluation package for machine translation performance benchmarking.
"""
from .evaluator import compute_metrics, evaluate_batch

__all__ = ["compute_metrics", "evaluate_batch"]
