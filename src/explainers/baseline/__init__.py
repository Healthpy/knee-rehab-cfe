"""
Baseline counterfactual explainers for XAI knee rehabilitation analysis.

This module contains the baseline explainers including MCELS and other
foundational methods that serve as comparison baselines.
"""

from .mcels_explainer import MCELSExplainer

__all__ = ['MCELSExplainer']