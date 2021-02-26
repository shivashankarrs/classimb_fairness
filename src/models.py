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

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def fit(self, X_train, y_train, X_val, y_val, optimizer, instance_weights=None, n_iter=100, batch_size=100, max_patience=10):
        X_val_tensor, y_val_tensor = variable(torch.FloatTensor(X_val)), y_val
        train_dataset = ImbDataset(X_train, y_train, instance_weights)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        best_score = -1
        best_state_dict = self.state_dict()
        patience = 0
        for i in range(n_iter):
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
    def __init__(self, X, y, instance_weights=None):
        super(ImbDataset, self).__init__()
        if instance_weights is None:
            instance_weights = np.ones(X.shape[0])
        self.X, self.y, self.instance_weights = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device), torch.FloatTensor(instance_weights).to(device) 
        
    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.instance_weights[index]
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), min(sample_size, len(self)))
        return [sample for sample in self.X[sample_idx]]


def train_epoch(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, criterion=F.cross_entropy):
    model.train()
    epoch_loss = 0
    for input, target, instance_weights in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        if isinstance(criterion, losses.CrossEntropyWithInstanceWeights) or isinstance(criterion, losses.LDAMLossInstanceWeight):
            loss = criterion(output, target, instance_weights)
        else:
            loss = criterion(output, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)