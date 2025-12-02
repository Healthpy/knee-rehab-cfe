"""Saliency base class."""
from abc import ABCMeta, abstractmethod


class Saliency(metaclass=ABCMeta):
    def __init__(self, background_data, background_label, predict_fn):
        self.background_data = background_data
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.perturbation_manager = None
        pass

    @abstractmethod
    def generate_saliency(self, data, label, **kwargs):
        pass