from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import numpy as np

import io
from PIL import Image


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, figsize=None, plot_figure=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if figsize is None:
        if len(classes) < 10:
            figsize = (7, 7)
        else:
            figsize = (len(classes)//2, len(classes)//2)
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if not plot_figure:
        backend = plt.get_backend()
        plt.switch_backend('agg')
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(-1, cm.shape[1]+1),
           yticks=np.arange(-1, cm.shape[0]+1),
           # ... and label them with the respective list entries
           xticklabels=[' '] + classes + [' '], yticklabels=[' '] + classes + [' '],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
       
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0], 3)
    buf.close()
    
    if not plot_figure:
        plt.switch_backend(backend)
        
    return image


def plot_matrix(pred1, pred2, classes1=None, classes2=None, plot_figure=True, title=' ', cmap=plt.cm.Blues,
                figsize=None):
    if figsize is None:
        if len(classes1) < 10:
            figsize = (7, 7)
        else:
            figsize = (len(classes1) // 2, len(classes2) // 2)

    if not plot_figure:
        backend = plt.get_backend()
        plt.switch_backend('agg')

    cm = np.zeros((len(classes1), len(classes2)), dtype=np.int32)
    for i, j in zip(pred2, pred1):
        cm[i, j] += 1
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(-1, cm.shape[1] + 1),
           yticks=np.arange(-1, cm.shape[0] + 1),
           # ... and label them with the respective list entries
           xticklabels=[' '] + classes2 + [' '], yticklabels=[' '] + classes1 + [' '],
           title=title,
           ylabel='class1',
           xlabel='class2')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0], 3)
    buf.close()

    if not plot_figure:
        plt.switch_backend(backend)

    return image