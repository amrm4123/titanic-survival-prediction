"""
Titanic Survival Prediction Package
----------------------------------
A machine learning package for predicting Titanic passenger survival.
"""

from .data_preprocessing import DataPreprocessor
from .model_training import TitanicModel
from .utils import (
    load_data,
    explore_data,
    plot_survival_distribution,
    save_submission
)

__version__ = '1.0.0'
__author__ = 'Amr AL-Kayal'

__all__ = [
    'DataPreprocessor',
    'TitanicModel',
    'load_data',
    'explore_data',
    'plot_survival_distribution',
    'save_submission'
]