import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from losses import NormedLinear
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

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
    def score(self, x, y_true):
        #accuracy
        return accuracy_score(y_true, self.predict(x))
    
    def get_hidden(self, x):
        x_var = variable(torch.FloatTensor(x))
        x_var = torch.tanh(self.fc1(x_var))
        return x_var.detach().cpu().numpy()

    def fit(self, X_train, y_train, X_val, y_val, optimizer, n_iter=100, batch_size=100, max_patience=10):
        X_val_tensor, y_val_tensor = variable(torch.FloatTensor(X_val)), y_val
        train_dataset = ImbDataset(X_train, y_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        best_score = -1
        best_state_dict = self.state_dict()
        patience = 0
        for i in range(n_iter):
            self.train()
            train_loss = train_epoch(self, optimizer, train_data_loader, self.criterion)            
            val_score = self.score(X_val_tensor, y_val_tensor)
            logging.info(f"iter {i} train loss {train_loss:.2f} val acc:{val_score:.2f}")
            if val_score > best_score:
                best_state_dict = self.state_dict()
                best_score = val_score
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
        self.load_state_dict(best_state_dict)

class LogReg(nn.Module):
    def __init__(self, input_size, output_size, normed_linear=False, criterion=F.cross_entropy):
        super(LogReg, self).__init__()
        self.fc1 = NormedLinear(input_size, output_size) if normed_linear else nn.Linear(input_size, output_size) 
        self.criterion = criterion
    def forward(self, x):
        x = self.fc1(x)
        return x
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            X_tensor = variable(torch.FloatTensor(x))
        else:
            X_tensor = x
        output = self.forward(X_tensor)
        y_pred = F.softmax(output, dim=1).max(dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    def score(self, x, y_true):
        #accuracy
        return accuracy_score(y_true, self.predict(x))
    
    def fit(self, X_train, y_train, X_val, y_val, optimizer, n_iter=100, batch_size=100, max_patience=10):
        X_val_tensor, y_val_tensor = variable(torch.FloatTensor(X_val)), y_val
        train_dataset = ImbDataset(X_train, y_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        best_score = -1
        best_state_dict = self.state_dict()
        patience = 0
        for i in range(n_iter):
            self.train()
            train_loss = train_epoch(self, optimizer, train_data_loader, self.criterion)            
            val_score = self.score(X_val_tensor, y_val_tensor)
            logging.info(f"train loss {train_loss:.2f} val acc:{val_score:.2f}")
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
    def __init__(self, X, y):
        super(ImbDataset, self).__init__()
        self.X, self.y = torch.FloatTensor(X), torch.LongTensor(y) 
        
    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), min(sample_size, len(self)))
        return [sample for sample in self.X[sample_idx]]


def train_epoch(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, criterion=F.cross_entropy):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)