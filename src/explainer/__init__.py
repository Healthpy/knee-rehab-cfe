"""
Explainer Module for Time-Series Data

Implements M-CELS (Counterfactual Explanation for Multivariate Time Series
Data Guided by Learned Saliency Maps) algorithm.

Reference:
    M-CELS: Counterfactual Explanation for Multivariate Time Series Data
    Guided by Learned Saliency Maps
    https://github.com/Luckilyeee/M-CELS
"""

from src.explainer.base import Saliency
from src.explainer.mcels_explainer import MCELSExplainer
from src.explainer.perturbation_manager import PerturbationManager
from src.explainer.shapley_adaptive_explainer import ShapleyAdaptiveExplainer
from src.explainer.learnable_gate_explainer import LearnableGateExplainer

__all__ = [
    'Saliency',
    'MCELSExplainer',
    'PerturbationManager',
    'ShapleyAdaptiveExplainer',
    'LearnableGateExplainer',
]
