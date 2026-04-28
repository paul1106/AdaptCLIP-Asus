"""Portable anomaly detection evaluation toolkit.

Copy this entire eval_toolkit/ directory to any anomaly detection repo.
Usage:
    from eval_toolkit.saver import PredictionSaver
    from eval_toolkit.metrics import AUROC, AUPR, AUPRO, F1Max
    # then run: python -m eval_toolkit.compute_metrics --manifest ... --output-dir ...
"""

from .saver import PredictionSaver

__all__ = ["PredictionSaver"]
