"""
Evaluation module for counterfactual explanations.

"""

from .metrics import (
    EvaluationMetrics,
    evaluate_single_result,
    evaluate_explanation_quality
)

__all__ = [
    'EvaluationMetrics',
    'evaluate_single_result', 
    'evaluate_explanation_quality'
]