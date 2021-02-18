import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from tqdm import tqdm

warnings.simplefilter("ignore")

from torch.utils.tensorboard import SummaryWriter 

from collections import defaultdict
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from utils.plotcm import plot_confusion_matrix

from torch.nn.utils.rnn import pad_sequence

from utils.metrics import Scorer


def collate(batch, q=95):
    if isinstance(batch[0], tuple):
        texts, targets = zip(*batch)
        targets = torch.LongTensor(targets)
    else:
        texts = batch
    length = np.percentile([len(x) for x in texts], interpolation='higher', q=q)
#    length = np.max([len(x) for x in texts])
    texts = [msg if len(msg) <= length else msg[:length] for msg in texts]
    texts = pad_sequence(texts, batch_first=True)
    
    if isinstance(batch[0], tuple):
        return texts, targets
    return texts


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def freeze(model):
    for param in model.parameters():
#        param.requires_grad = True
        param.requires_grad = False
    for param in model.bert.parameters():
        param.requires_grad = False
#    for param in model.bert.cls.parameters():
#        param.requires_grad = True

    for param in model.out.parameters():
        param.requires_grad = True

    for param in model.out2.parameters():
        param.requires_grad = True

    model.layer_weights.requires_grad = True

    model.layer_weights2.requires_grad = True

    return model


def write_text(writer, model_name, history, split=0, epoch=0, batch=None):
    text_with_metrics = ''
    for metric in history:
        text_with_metrics += metric + ': ___'+ str(history[metric]) + '___;   '
    if batch is None:
        writer.add_text(model_name + '/split_' + str(split) + '/epoch_' + str(epoch), text_with_metrics, batch)
    else:
        writer.add_text(model_name + '/split ' + str(split), text_with_metrics, epoch)


def write_scalars(writer, model_name, history, n_epoch, epoch):
#    writer.add_scalar(model_name + ' non-binary/loss/train total', history['loss_train'], split * n_epoch + epoch)
#    writer_split.add_scalar(model_name + ' non-binary/loss/train split ' + str(split), history['loss_train'], epoch)

#    writer.add_scalar(model_name + ' non-binary/loss/val total', history['loss_val'], split * n_epoch + epoch)
#    writer_split.add_scalar(model_name + ' non-binary/loss/val split ' + str(split), history['loss_val'], epoch)

    writer.add_scalars(model_name + ' non-binary/loss total', {'train': history['loss_train'],\
                                                               'val': history['loss_val']}, epoch)
#    writer_split.add_scalars(model_name + ' non-binary/loss split ' + str(split), {'train': history['loss_train'],\
#                                                                                   'val': history['loss_val']}, epoch)
    for metric in history:
        if 'loss' not in metric and '@' not in metric and 'roc_auc' not in metric:
            writer.add_scalar(f'{model_name}_binary/{metric}/total', history[metric], epoch)
#            writer_split.add_scalar(f'{model_name}_binary/{metric}/split_{split}', history[metric], epoch)
        else:
            writer.add_scalar(f'{model_name}_non-binary/{metric}/total', history[metric], epoch)
#            writer_split.add_scalar(f'{model_name}_non-binary/{metric}/split_{split}', history[metric], epoch)


def print_results(model_name, history, epoch=None):
    print(model_name)
    if epoch is not None:
        print(f'epoch: {epoch}   train loss: {history["loss_train"]}   val loss: {history["loss_val"]}')
    for metric in history:
        if 'loss' not in metric:
            print(f'{metric}: {history[metric]}')

            
def calc_weights(train_labels, n_classes, device):    
    clss, cnts = np.unique(train_labels, return_counts=True)
    tmp = np.ones(n_classes, dtype=np.int64)
    for cls, cnt in zip(clss, cnts):
        tmp[cls] = cnt
    weights = len(train_labels)/(n_classes*tmp)
    return torch.FloatTensor(weights).to(device)


def stable_softmax(x):
    z = x - np.max(x, axis=1).reshape(-1, 1)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)
    softmax = numerator/denominator
    return softmax
            
#pred_proba_test = np.exp(pred_proba_test)/np.sum(np.exp(pred_proba_test), axis=1).reshape(-1, 1)
    
            
def copy_model(model, device):
    model.to('cpu')
    new_model = copy.deepcopy(model)
    model.to(device)
    return new_model

            
class EarlyStop():
    """
        if patience <= 0 then object will always return False and will only save the best model 
    """ 
    def __init__(self, patience=0):
        self.best_model = None
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        
    def __call__(self, metric, model, device):
        if self.best_metric is None:
            self.best_model = copy_model(model, device)
            self.best_metric = metric
            return False
        if metric > self.best_metric:
                self.counter = 0
                self.best_model = copy_model(model, device)
                self.best_metric = metric
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
            
            
class NNTrainer():
    """
    метод train - обучение сетей и сбор статистики по метрикам на этапе обучения
    метод test - inference модели и подсчет метрик, если передаются размеченные данные
    """
    def __init__(self, model, device='cpu', label_names=None, model_name='model', bert=False):
        self.model = model
        self.model_name = model_name
        self.label_names = label_names
#        self.batch_size = batch_size
        self.device = device
        self.loss_train = defaultdict(list)
        self.loss_val = defaultdict(list)
        self.accuracy = defaultdict(list)
        self.precision_micro = defaultdict(list)
        self.precision_macro = defaultdict(list)
        self.recall_micro = defaultdict(list)
        self.recall_macro = defaultdict(list)
        self.roc_auc_micro = defaultdict(list)
        self.roc_auc_macro = defaultdict(list)
        self.f1_micro = defaultdict(list)
        self.f1_macro = defaultdict(list)
        self.loss = None

        
    def eval_step(self, batch):
        y_batch = None
        with torch.no_grad():
            if isinstance(batch, tuple) or isinstance(batch, list):
                x_batch, y_batch = batch
                if isinstance(x_batch, list):
                    length = len(x_batch)
                    batch_pred = self.model(x_batch[0].to(self.device))
                    for x in x_batch[1:]:
                        batch_pred += self.model(x.to(self.device))
                    batch_pred /= length
                else:
                    batch_pred = self.model(x_batch.to(self.device))
            else:
                batch_pred = self.model(batch.to(self.device))
        return batch_pred, y_batch


    def train(self, train_loader, val_loader, extra_loader,\
                  n_epoch, optim=Adam, weight_decay=0., schedul=None, loss=torch.nn.CrossEntropyLoss,\
                  weighted=True, lr=2e-5, accum_steps=1, show_results=True, saved_models_dir=None, verbose=True, patience=0,\
                  eta=0.1, freeze_epoch=0, one_split=True, calc_metric_batch=None, stop_metric='accuracy', alpha=0.8, betha=0.2):
        params = locals()
        scorer = Scorer()
        early_stop = EarlyStop(patience)
        writer = SummaryWriter(comment='_'+self.model_name + '_total_', flush_secs=600, max_queue=100)
#        total_history = defaultdict(list)

        param_text = ''
        for param in params:
            if param != 'train_data' and param != 'train_labels' and param !='self':
                param_text += param + ': ___'+ str(params[param]) + '___;   '
        writer.add_text(self.model_name + ' train parameters ', param_text)
            
        self.model.to(self.device)
        
        n_classes = len(self.label_names)

        if weighted and not self.label_names is None:
            self.loss = loss(weight=calc_weights(train_loader.dataset.labels, n_classes, self.device))
        else:
            self.loss = loss

        optimizer = optim(self.model.parameters(), weight_decay=weight_decay, lr=lr)
        
        if not schedul is None:
            scheduler = schedul(optimizer)
            
        if extra_loader is not None:
            extra_iter = iter(extra_loader) 

        for epoch in range(n_epoch):
            loss_train_lst = []
            
            self.model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm_notebook(train_loader, disable=not verbose)):                
                batch_pred = self.model(x_batch.to(self.device))
                batch_loss = self.loss(batch_pred, y_batch.to(self.device))
                
                if extra_loader is not None:
                    try:
                        batch_with_aug, batch_without_aug = next(extra_iter)
                    except StopIteration:
                        extra_iter = iter(extra_loader)
                        batch_with_aug, batch_without_aug = next(extra_iter)
                    self.model.eval()
                    with torch.no_grad():
                        pseudo_labels = torch.argmax(self.model(batch_without_aug.to(self.device)), dim=1)
                        
                    self.model.train()
                    extra_batch_pred = self.model(batch_with_aug.to(self.device))
                    extra_batch_loss = self.loss(extra_batch_pred, pseudo_labels)
                    batch_loss = alpha*batch_loss + betha*extra_batch_loss    
                    
                
                loss_train_lst.append(batch_loss.item())
                    
                writer.add_scalar(self.model_name + ' non-binary/batch loss/train split ' + ' epoch ' + str(epoch), loss_train_lst[-1], batch_idx)
                batch_loss.backward()
                if (batch_idx + 1) % accum_steps == 0:
                    if not schedul is None:
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                
                if (not calc_metric_batch is None) and ((batch_idx + 1)%calc_metric_batch == 0):
                    lbl_val_lst = []
                    pred_val_lst = []
                    pred_proba_val_lst =[]
                    loss_val_lst = []
                    
                    self.model.eval()
                    for x_batch, y_batch in tqdm_notebook(val_loader, disable=not verbose):
                        with torch.no_grad():
                            batch_pred = self.model(x_batch.to(self.device))
                            batch_loss = self.loss(batch_pred, y_batch.to(self.device))
                            loss_val_lst.append(batch_loss.item())
                            lbl_val_lst.append(y_batch.numpy().reshape(-1, 1))

                        pred_val_lst.append(torch.argmax(batch_pred, dim=1).cpu().reshape(-1, 1))
                        pred_proba_val_lst.append(batch_pred.cpu().numpy())

                    lbl_val = np.vstack(lbl_val_lst)

                    pred_val = np.vstack(pred_val_lst)
                    pred_proba_val = np.vstack(pred_proba_val_lst)

                    history_medium = scorer(pred_val, pred_proba_val, lbl_val)
                    history_medium['loss_val'] = np.mean(loss_val_lst)

                    write_text(writer, self.model_name, history_medium, split=split_id, epoch=epoch, batch=batch_idx)

                    if not saved_models_dir is None:    
                        torch.save(self.model.state_dict(), saved_models_dir + self.model_name + '_' + '_' + str(epoch) + '_' + str(batch_idx) + '.pth')
                            
                    if early_stop(history_medium[stop_metric], self.model, self.device):
                        break
                    self.model.train()
                
            lbl_val_lst = []
            pred_val_lst = []
            pred_proba_val_lst =[]
            loss_val_lst = []

            self.model.eval()
            for batch_idx, batch in enumerate(tqdm_notebook(val_loader, disable=not verbose)):
#                with torch.no_grad():
                
                batch_pred, y_batch = self.eval_step(batch)    
#                    batch_pred = self.model(x_batch.to(self.device))
                    
                batch_loss = self.loss(batch_pred, y_batch.to(self.device))
                        
                loss_val_lst.append(batch_loss.item())
                        
                pred_val_lst.append(torch.argmax(batch_pred, dim=1).cpu().reshape(-1, 1))
                pred_proba_val_lst.append(batch_pred.cpu().numpy())

                lbl_val_lst.append(y_batch.numpy().reshape(-1, 1))

            lbl_val = np.vstack(lbl_val_lst)

            pred_val = np.vstack(pred_val_lst)
            pred_proba_val = np.vstack(pred_proba_val_lst)

            history = scorer(pred_val, pred_proba_val, lbl_val)
            history['loss_train'] = np.mean(loss_train_lst)
            history['loss_val'] = np.mean(loss_val_lst)
#                total_history[split_id].append(history)
            write_scalars(writer, self.model_name, history, n_epoch=n_epoch, epoch=epoch)
            write_text(writer, self.model_name, history, epoch=epoch)
        
            if show_results:
                print_results(self.model_name, history, epoch)
                cm_image_n = plot_confusion_matrix(lbl_val, pred_val, classes=self.label_names, normalize=True, title='Normalized confusion matrix')
                cm_image = plot_confusion_matrix(lbl_val, pred_val, classes=self.label_names, normalize=False, title='Confusion matrix')
                plt.show()
            else:
                cm_image_n = plot_confusion_matrix(lbl_val, pred_val, classes=self.label_names, normalize=True, title='Normalized confusion matrix', plot_figure=False)
                cm_image = plot_confusion_matrix(lbl_val, pred_val, classes=self.label_names, normalize=False, title='Confusion matrix', plot_figure=False)

            writer.add_image(self.model_name + ' Normalized confusion matrix split ', cm_image_n, epoch, dataformats='HWC')
            writer.add_image(self.model_name + ' Confusion matrix split ', cm_image, epoch, dataformats='HWC')

            if early_stop(history[stop_metric], self.model, self.device):
                break
        
            if not saved_models_dir is None:    
                torch.save(self.model.state_dict(), saved_models_dir + self.model_name + '_' + str(epoch) + '.pth')    

        if not saved_models_dir is None:
            torch.save(early_stop.best_model.state_dict(), saved_models_dir + self.model_name + '_best_' + '.pth')
        writer.close()
        return early_stop.best_metric
    
    
    def test(self, test_loader, tta=False, show_results=True, verbose=True, log=True):
        if log:
            writer = SummaryWriter(comment='_'+self.model_name + '_test', max_queue=50)
        scorer = Scorer()

        loss_test_lst=[]
        lbl_test_lst = []
        pred_test_lst = []
        pred_proba_test_lst =[]

        self.model.to(self.device)
        self.model.eval()
        
        y_batch = None
        
        for batch_idx, batch in enumerate(tqdm_notebook(test_loader, disable=not verbose)):    
                batch_pred, y_batch = self.eval_step(batch)

                if y_batch is not None:
                    lbl_test_lst.append(y_batch.numpy().reshape(-1,1))

                pred_test_lst.append(torch.argmax(batch_pred, dim=1).cpu().reshape(-1, 1))
                pred_proba_test_lst.append(batch_pred.cpu().numpy())

        if y_batch is not None:
            lbl_test = test_loader.dataset.labels

        pred_test = np.vstack(pred_test_lst)
        pred_proba_test = np.vstack(pred_proba_test_lst)
        
        pred_proba_test = stable_softmax(pred_proba_test)
        
#        pred_proba_test = np.exp(pred_proba_test)/np.sum(np.exp(pred_proba_test), axis=1).reshape(-1, 1)

        if y_batch is not None:
            history = scorer(pred_test, pred_proba_test, lbl_test)
            if show_results:
                print_results(model_name=self.model_name, history=history)
                cm_image_n = plot_confusion_matrix(lbl_test, pred_test, classes=self.label_names, normalize=True, title='Normalized confusion matrix')
                cm_image = plot_confusion_matrix(lbl_test, pred_test, classes=self.label_names, normalize=False, title='Confusion matrix')
                plt.show()
            else:
                cm_image_n = plot_confusion_matrix(lbl_test, pred_test, classes=self.label_names, normalize=True, title='Normalized confusion matrix', plot_figure=False)
                cm_image = plot_confusion_matrix(lbl_test, pred_test, classes=self.label_names, normalize=False, title='Confusion matrix', plot_figure=False)
            if log:
                write_text(writer, self.model_name + '_test', history)
                writer.add_image(self.model_name + ' test Normalized confusion matrix', cm_image_n, dataformats='HWC')
                writer.add_image(self.model_name + ' test Confusion matrix', cm_image, dataformats='HWC')
                writer.close()
            return pred_test, pred_proba_test, history
        return pred_test, pred_proba_test