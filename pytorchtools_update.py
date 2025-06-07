import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', mode='min', monitor='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        #self.val_loss_min = np.Inf
        if self.mode == 'min':
            self.val_best = np.Inf
        else:
            self.val_best = -np.Inf

    def __call__(self, val_metric, model):
        score = val_metric
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when performance increase.'''
        if self.verbose:
            if self.mode == 'min':
                print(f'Validation loss decreased ({self.val_best:.6f} --> {val_metric:.6f}).  Saving model ...')
            else:
                #print(f'Validation R-squre increased ({self.val_best:.6f} --> {val_metric:.6f}).  Saving model ...')
                print(f'Validation {self.monitor} increased ({self.val_best:.6f} --> {val_metric:.6f}).  Saving model ...')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), self.path)  # 如果使用多 GPU，保存 model.module
        else:
            torch.save(model.state_dict(), self.path)  # 单 GPU 或 CPU
            #torch.save(model.state_dict(), self.path)
        self.val_best = val_metric
