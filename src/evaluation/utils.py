import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def classification_metrics(y_true, y_pred, labels=None):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return report


def plot_confusion(y_true, y_pred, labels=None, figsize=(6, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    return fig, ax
