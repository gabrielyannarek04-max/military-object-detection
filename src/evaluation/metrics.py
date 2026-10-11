import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_with_background(y_true, y_pred, classes, title="Confusion Matrix", cmap="Blues"):
    """
    Computes and plots a confusion matrix ensuring the 'Background' class is properly handled
    and highlighted for false detections on background clutter.
    
    Args:
        y_true (array-like): Ground truth labels (including background).
        y_pred (array-like): Predicted labels (including background).
        classes (list of str): Class names, e.g., ['Background', 'Class1', 'Class2', ...]
        title (str): Title of the plot.
        cmap (str): Colormap for the heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    
    # Normalize confusion matrix across true classes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=cm, fmt="d", cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    
    plt.title(title, fontsize=16, pad=15)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Highlight the background class (assuming it's index 0)
    if 'Background' in classes:
        bg_idx = classes.index('Background')
        plt.gca().get_xticklabels()[bg_idx].set_color("red")
        plt.gca().get_yticklabels()[bg_idx].set_color("red")
        plt.gca().get_xticklabels()[bg_idx].set_weight("bold")
        plt.gca().get_yticklabels()[bg_idx].set_weight("bold")
        
    plt.tight_layout()
    plt.show()
