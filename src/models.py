import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from losses import NormedLinear
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import logging
import losses
import pdb
from copy import deepcopy

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fair_reg(preds, Xp):
    Xp = torch.cat((Xp, 1-Xp), dim=1)
    viol = preds.mean()-(preds@Xp)/torch.max(Xp.sum(axis=0), torch.ones(Xp.shape[1])*1e-5)
    return (viol**2).mean()

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, normed_linear=False, criterion=F.cross_entropy):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = NormedLinear(hidden_size, output_size) if normed_linear else nn.Linear(hidden_size, output_size) 
        self.criterion = criterion
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    def predict(self, x):
        if not isinstance(x, Variable):
            X_tensor = variable(torch.FloatTensor(x))
        else:
            X_tensor = x
        output = self.forward(X_tensor)
        y_pred = F.softmax(output, dim=1).max(dim=1)[1]
        y_pred = y_pred.detach().cpu().numpy()
        return y_pred
    def score(self, x, y_true, metric='f1'):
        #metric can be 'f1' (macro f1) or acc (accuracy)
        if metric == 'f1':
            return f1_score(y_true, self.predict(x), average='macro')
        elif metric == 'acc':
        #accuracy
            return accuracy_score(y_true, self.predict(x))
    
    def get_hidden(self, x):
        x_var = variable(torch.FloatTensor(x))
        x_var = torch.tanh(self.fc1(x_var))
        return x_var.detach().cpu().numpy()

    def fit(self, X_train, y_train, y_p_train, y_mp_train, X_val, y_val, optimizer, instance_weights=None, n_iter=100, batch_size=100, max_patience=10):
        X_val_tensor, y_val_tensor = variable(torch.FloatTensor(X_val)), y_val
        train_dataset = ImbDataset(X_train, y_train, y_p_train, y_mp_train, instance_weights)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        best_score = -1
        best_state_dict = self.state_dict()
        patience = 0
        init_lr = optimizer.defaults['lr']
        #delayed reweighting DRW start after epoch delayed_epoch, don't delay if delayed_epoch = -1
        class_weight = None
        delayed_epoch = -1
        if hasattr(self.criterion, 'class_weight'):
            class_weight = deepcopy(self.criterion.class_weight)
            self.criterion.class_weight = None



        for i in range(n_iter):
            #for use with sgd
            #adjust_learning_rate(optimizer, i, init_lr)
            if i > delayed_epoch and class_weight is not None:
                self.criterion.class_weight = class_weight
            self.train()
            train_loss = train_epoch(self, optimizer, train_data_loader, self.criterion)            
            val_score = self.score(X_val_tensor, y_val_tensor)
            logging.debug(f"iter {i} train loss {train_loss:.2f} val f1:{val_score:.2f}")
            if val_score > best_score:
                best_state_dict = self.state_dict()
                best_score = val_score
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
        self.load_state_dict(best_state_dict)


class ImbDataset(Dataset):
    """load a dataset"""
    def __init__(self, X, y, y_p=None, y_mp=None, instance_weights=None):
        super(ImbDataset, self).__init__()
        if instance_weights is None:
            instance_weights = np.ones(X.shape[0])
        if y_p is None:
            y_p = np.ones(X.shape[0])
        if y_mp is None:
            y_mp = np.ones(X.shape[0])
        self.X, self.y, self.y_p, self.y_mp, self.instance_weights = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device), torch.LongTensor(y_p).to(device), torch.LongTensor(y_mp).to(device), torch.FloatTensor(instance_weights).to(device) 
        
    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.y_p[index], self.y_mp[index], self.instance_weights[index]
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), min(sample_size, len(self)))
        return [sample for sample in self.X[sample_idx]]


def train_epoch(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, criterion=F.cross_entropy, regularize = False):
    model.train()
    epoch_loss = 0
    for input, target, group, targetgroup, instance_weights in data_loader:
        input, target, group, targetgroup = variable(input), variable(target), variable(group), variable(targetgroup)
        optimizer.zero_grad()
        output = model(input)
        if isinstance(criterion, losses.CrossEntropyWithInstanceWeights) or isinstance(criterion, losses.LDAMLossInstanceWeight):
            loss = criterion(output, target, instance_weights)
        elif isinstance(criterion, losses.GLDAMLoss):
            loss = criterion(output, target, group)
        elif isinstance(criterion, losses.GeneralLDAMLoss):
            loss = criterion(output, target, group, targetgroup, instance_weights)
        else:
            loss = criterion(output, target)
        epoch_loss += loss.data
        if regularize:
            reg = fair_reg(output[target==1], group[target==1])
            epoch_loss += reg
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda(device)
    return Variable(t, **kwargs)


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = init_lr * epoch / 5
    elif epoch > 17:
        lr = init_lr * 0.0001
    elif epoch > 10:
        lr = init_lr * 0.01
    else:
        lr = init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr