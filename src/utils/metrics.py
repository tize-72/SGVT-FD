"""
Evaluation metrics for fault diagnosis.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report


def compute_metrics(y_true, y_pred, average='macro'):
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        Dictionary with accuracy, f1, precision, recall
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


def compute_confusion_matrix(y_true, y_pred, class_names=None):
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        confusion_matrix: numpy array
        report: Classification report string
    """
    cm = sk_confusion_matrix(y_true, y_pred)
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    else:
        report = classification_report(y_true, y_pred, zero_division=0)
    return cm, report


def per_class_accuracy(y_true, y_pred, class_names):
    """Compute per-class accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary mapping class name to accuracy
    """
    cm = sk_confusion_matrix(y_true, y_pred)
    per_class = {}
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class[name] = cm[i, i] / cm[i].sum()
        else:
            per_class[name] = 0.0
    return per_class
