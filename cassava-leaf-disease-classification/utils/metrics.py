import pandas as pd
import numpy as np
import copy
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def rprecision_at_k(pred_proba, labels, k=10):
    labels_k = np.argsort(-pred_proba, axis=1)[:,:k]
    return np.sum([int(lbl in lbls) for lbls, lbl in zip(labels_k, labels)])/len(labels)


def vectorize(preds, y, k):
    labels = np.array(
        list(
            map(
                lambda x: [int(el in x[1]) for el in x[0]], 
                list(zip((-preds).argsort(axis=-1)[:, :k], y))
            )
        )
    )
    return labels


def ap_at_k(pred_proba, labels, k=10):
    lbls = vectorize(pred_proba, labels.reshape(-1, 1), k)
    labels_k = lbls[:, :k]

    mask = labels_k.sum(axis=1) == 0
        
    tmp = (((np.cumsum(labels_k[~mask], axis=-1) * labels_k[~mask]) / np.arange(1, k + 1)).sum(axis=1)\
                   / np.minimum(lbls[~mask].sum(axis=1), k))
        
    return tmp.sum() / lbls.shape[0]


def calculate_metrics(pred_proba, test_labels, label_names, k=10, th=0.5, softmax=True, round_values=3, sort_by=['precision', 'recall']):
    if softmax:
        pred_proba = F.softmax(torch.from_numpy(pred_proba), dim=1).numpy()

    probas = {label: pred_proba[:, i] for i, label in enumerate(label_names)}
    labels = {label: (probas[label] > th).astype(int) for label in label_names}
    true_labels = {label: (test_labels == i).astype(int) for i, label in enumerate(label_names)}
    
    metrics = defaultdict(list)
    for label in label_names:
        metrics['precision'].append(precision_score(true_labels[label], labels[label]))
        metrics['recall'].append(recall_score(true_labels[label], labels[label]))
        metrics['accuracy'].append(accuracy_score(true_labels[label], labels[label]))
        metrics['f1'].append(f1_score(true_labels[label], labels[label]))
        metrics['roc_auc'].append(roc_auc_score(true_labels[label], probas[label]))
    for i in range(1, k + 1, 1):
        metrics['rpecision@k'].append(rprecision_at_k(pred_proba, test_labels, i))
        metrics['ap@k'].append(ap_at_k(pred_proba, test_labels, i))
        
    metrics_df = pd.DataFrame(label_names, columns=['category'])
    
    metrics_df['precision'] = metrics['precision']
    metrics_df['recall'] = metrics['recall']
    metrics_df['accuracy'] = metrics['accuracy']
    metrics_df['f1'] = metrics['f1']
    metrics_df['roc_auc'] = metrics['roc_auc']
 
    metrics2_df = pd.DataFrame(list(range(1, k + 1, 1)), columns=['k'])
    metrics2_df['rpecision@k'] = metrics['rpecision@k'] 
    metrics2_df['ap@k'] = metrics['ap@k']
    
    metrics_df.set_index('category', inplace=True)
    
    if round_values > 0:
        for column in metrics_df.columns:
            metrics_df[column] = metrics_df[column].apply(lambda x: np.round(x, round_values))
    
    return metrics_df.sort_values(by=sort_by, ascending=False), metrics2_df.set_index('k')


class Scorer:
    def __init__(self):
        self.binary_scorers = {'f1_micro': (lambda x, y: f1_score(x, y, average='micro')),
                               'f1_macro': (lambda x, y: f1_score(x, y, average='macro')),
                               'precision_micro': (lambda x, y: precision_score(x, y, average='micro')),
                               'precision_macro': (lambda x, y: precision_score(x, y, average='macro')),
                               'recall_micro': (lambda x, y: recall_score(x, y, average='micro')),
                               'recall_macro': (lambda x, y: recall_score(x, y, average='macro')),
                               'accuracy': accuracy_score}
        self.non_binary_scorers = {'roc_auc_micro': (lambda x, y: roc_auc_score(x, y, average='micro')),
                                   'roc_auc_macro': (lambda x, y: roc_auc_score(x, y, average='macro'))}

        self.non_binary_scorers['rprecision@1'] = (lambda x, y: rprecision_at_k(y, x, k=1))
        self.non_binary_scorers['rprecision@2'] = (lambda x, y: rprecision_at_k(y, x, k=2))
        self.non_binary_scorers['rprecision@3'] = (lambda x, y: rprecision_at_k(y, x, k=3))
        self.non_binary_scorers['rprecision@4'] = (lambda x, y: rprecision_at_k(y, x, k=4))
#         self.non_binary_scorers['rprecision@5'] = (lambda x, y: rprecision_at_k(y, x, k=5))
#         self.non_binary_scorers['rprecision@6'] = (lambda x, y: rprecision_at_k(y, x, k=6))
#         self.non_binary_scorers['rprecision@7'] = (lambda x, y: rprecision_at_k(y, x, k=7))
#         self.non_binary_scorers['rprecision@8'] = (lambda x, y: rprecision_at_k(y, x, k=8))
#         self.non_binary_scorers['rprecision@9'] = (lambda x, y: rprecision_at_k(y, x, k=9))
#         self.non_binary_scorers['rprecision@10'] = (lambda x, y: rprecision_at_k(y, x, k=10))

        self.non_binary_scorers['ap@1'] = (lambda x, y: ap_at_k(y, x, k=1))
        self.non_binary_scorers['ap@2'] = (lambda x, y: ap_at_k(y, x, k=2))
        self.non_binary_scorers['ap@3'] = (lambda x, y: ap_at_k(y, x, k=3))
        self.non_binary_scorers['ap@4'] = (lambda x, y: ap_at_k(y, x, k=4))
#         self.non_binary_scorers['ap@5'] = (lambda x, y: ap_at_k(y, x, k=5))
#         self.non_binary_scorers['ap@6'] = (lambda x, y: ap_at_k(y, x, k=6))
#         self.non_binary_scorers['ap@7'] = (lambda x, y: ap_at_k(y, x, k=7))
#         self.non_binary_scorers['ap@8'] = (lambda x, y: ap_at_k(y, x, k=8))
#         self.non_binary_scorers['ap@9'] = (lambda x, y: ap_at_k(y, x, k=9))
#         self.non_binary_scorers['ap@10'] = (lambda x, y: ap_at_k(y, x, k=10))
#        for i in range(1, k + 1, 1):
#            self.non_binary_scorers['rprecision@'+str(i)] = (lambda x, y: rprecision_at_k(y, x, k=i))
#        for i in range(1, k + 1, 1):
#            self.non_binary_scorers['ap@'+str(i)] = (lambda x, y: ap_at_k(y, x, k=i))

    def __call__(self, pred, pred_proba, labels, round=3):
        n_classes = pred_proba.shape[1]
        ohe_labels = np.zeros((labels.shape[0], n_classes))
        for i, label in enumerate(labels):
            ohe_labels[i, label] = 1
        history = defaultdict(list)
        for scorer_name, scorer in self.binary_scorers.items():
            history[scorer_name] = np.round(scorer(labels, pred), round)
        for scorer_name, scorer in self.non_binary_scorers.items():
            if 'roc_auc' in scorer_name:
                if scorer_name is 'roc_auc_macro':
                    try:
                        history[scorer_name] = np.round(scorer(ohe_labels, pred_proba), round)
                    except ValueError:
                        history[scorer_name] = 0
                else:
                    history[scorer_name] = np.round(scorer(ohe_labels, pred_proba), round)
            else:
                history[scorer_name] = np.round(scorer(labels, pred_proba), round)
        return history
