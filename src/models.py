import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from losses import NormedLinear


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
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        output = self.forward(X_tensor)
        y_pred = F.softmax(output, dim=1).max(dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    def score(self, X, y_true):
        #accuracy
        return accuracy_score(y_true, self.predict(X))
    
    def fit(self, X_train, y_train, X_val, y_val, optimizer, n_iter=100):
        X_train_tensor, X_val_tensor = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
        y_train_tensor, y_val_tensor = torch.LongTensor(y_train), torch.LongTensor(y_val)

        best_score = -1
        best_state_dict = self.state_dict()
        for i in range(n_iter):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X_train_tensor)
            loss = self.criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            val_score = self.score(X_val_tensor, y_val_tensor)
            if val_score > best_score:
                best_state_dict = self.state_dict()
                best_score = val_score
        self.load_state_dict(best_state_dict)

class LogReg(nn.Module):
    def __init__(self, input_size, output_size, normed_linear=False, criterion=F.cross_entropy):
        super(LogReg, self).__init__()
        self.fc1 = NormedLinear(input_size, output_size) if normed_linear else nn.Linear(input_size, output_size) 
        self.criterion = criterion
    def forward(self, x):
        x = self.fc1(x)
        return x
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        output = self.forward(X_tensor)
        y_pred = F.softmax(output, dim=1).max(dim=1)[1]
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    def score(self, X, y_true):
        #accuracy
        return accuracy_score(y_true, self.predict(X))
    
    def fit(self, X_train, y_train, X_val, y_val, optimizer, n_iter=100):
        X_train_tensor, X_val_tensor = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
        y_train_tensor, y_val_tensor = torch.LongTensor(y_train), torch.LongTensor(y_val)

        best_score = -1
        best_state_dict = self.state_dict()
        for i in range(n_iter):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X_train_tensor)
            loss = self.criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            val_score = self.score(X_val_tensor, y_val_tensor)
            if val_score > best_score:
                best_state_dict = self.state_dict()
                best_score = val_score
        self.load_state_dict(best_state_dict)


'''
class DeepMojiModel(Model):
    """
    simple MLP model, that operates over encoded states
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    emb_size: ``int``, required
        the size of the input embedding
    hidden_size: ``int``, required
        the size of the hidden layer
    """

    def __init__(self,
                 vocab: Vocabulary,
                 emb_size: int,
                 hidden_size: int,
                 ) -> None:
        super().__init__(vocab)
        self.emb_size = emb_size

        # an mlp with one hidden layer
        layers = []
        layers.append(nn.Linear(self.emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.scorer = nn.Sequential(*layers)

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}

    
    def forward(self,
                vec: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        vec: torch.Tensor, required
            The input vector.
        label: torch.Tensor, optional (default = None)
            A variable of the correct label.
        Returns
        -------
        An output dictionary consisting of:
        y_hat: torch.FloatTensor
            the predicted values
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        scores = self.scorer(vec)
        y_hat = torch.argmax(scores, dim=1)

        output = {"y_hat": y_hat}
        if label is not None:
            self.metrics['accuracy'](y_hat, label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), label)
            output["loss"] = self.loss(scores, label)

        return output

    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "p": p,
                "r": r,
                "f1": f1}      
'''