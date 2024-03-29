import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

'''
focal loss and ldam loss taken from https://github.com/kaidic/LDAM-DRW
self adjusted dice loss taken from https://github.com/fursovia/self-adj-dice/ based on Dice Loss for Data-imbalanced NLP Tasks
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fair_reg(preds, Xp):
    Xp = Xp.unsqueeze(1)
    Xp = Xp.type(torch.FloatTensor).to(device)
    Xp = torch.cat((Xp, 1-Xp), dim=1)
    viol = preds.mean()-(preds@Xp)/torch.max(Xp.sum(axis=0), torch.ones(Xp.shape[1]).to(device)*1e-5)
    return (viol**2).mean()


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out



class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        '''
        cls_num_list: the number of instances in each class, a list of numbers where cls_num_list[0] is the number of instances in class 0
        weight: a vector weight of each class (can be different from |C_i| / sum(|C_j|)) as in  Class-balanced loss based on effective
        number of samples implemented in RW in LDAM-DRW where weights are:
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
            note: for ldam loss the last layer of the model should NOT be nn.Linear, it should be nn.NormedLinear
        '''
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8).to(device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class GeneralLDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, clsp_num_list, mp_num_list, max_m=0.5, class_weight=None, ldams=30, ldamc=0.5, ldamg=0.5, ldamcg=0.5, use_instance=False, ldam_mul_c_g=False, rho=0.0):
        '''
        cls_num_list: the number of instances in each class, a list of numbers where cls_num_list[0] is the number of instances in class 0
        clsp_num_list: the number of instances in each group
        weight: a vector weight of each class (can be different from |C_i| / sum(|C_j|)) as in  Class-balanced loss based on effective
        number of samples implemented in RW in LDAM-DRW where weights are:
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
            note: for ldam loss the last layer of the model should NOT be nn.Linear, it should be nn.NormedLinear
        
        crossentropy normal: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=1, ldamc=0, ldamg=0, use_instance=False)
        crossentropy with class weights: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=1, ldamc=0, ldamg=0, use_instance=False)
        crossentropy with instance weights: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=1, ldamc=0, ldamg=0, use_instance=True)
        crossentropy with class weight and instance weight GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=1, ldamc=0, ldamg=0, use_instance=True)
        
        ldam c: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=30, , ldamc=1, ldamg=0, use_instance=False)
        ldam c with instance weight: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=30, ldamc=1, ldamg=0, use_instance=True)
        ldam c with class weight: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=30, ldamc=1, ldamg=0, use_instance=False)
        ldam c with instance and class weight: GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=30, ldamc=1, ldamg=0, use_instance=True)
        ldam g GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=30, ldamc=0, ldamg=1, use_instance=False)
        ldam cg GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=None, ldams=30, ldamc=0.5, ldamg=0.5, use_instance=False)
        ldam cg with class weights GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=30, ldamc=0.5, ldamg=0.5, use_instance=False)
        ldam cg with class weights and instance weight GeneralLDAMLoss(cls_num_list, clsp_num_list, max_m=0.5, class_weight=***, ldams=30, ldamc=0.5, ldamg=0.5, use_instance=True)
        '''
        super(GeneralLDAMLoss, self).__init__()
        self.max_m = max_m
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)

        #do the same for groups or protected labels
        g_list = 1.0 / np.sqrt(np.sqrt(clsp_num_list))
        g_list = g_list * (self.max_m / np.max(g_list))
        g_list = torch.FloatTensor(g_list).to(device)


        #mg_list = 1.0 / np.sqrt(np.sqrt(mp_num_list))
        mg_list = 1.0 / np.power(mp_num_list, 0.25)
        mg_list = mg_list * (self.max_m / np.max(mg_list))
        mg_list = torch.FloatTensor(mg_list).to(device)

        self.g_list = g_list
        self.m_list = m_list
        self.mg_list = mg_list

        assert ldams > 0
        #assert 0 <= g <= 1
        self.ldamg = ldamg
        self.ldams = ldams
        self.ldamc = ldamc
        self.ldamcg = ldamcg
        self.use_instance = use_instance
        self.class_weight = class_weight
        self.ldam_mul_c_g = ldam_mul_c_g
        self.ce_instance = CrossEntropyWithInstanceWeights(class_weights=self.class_weight)
        self.rho = rho

    def __repr__(self):
        name = ['ldam']
        name.append('iw' if self.use_instance else 'noiw')
        name.append(f"cg{self.ldamcg}")
        name.append(f"s{self.ldams}")
        name.append(f"maxm{self.max_m}")
        name.append('cw' if self.class_weight is not None else 'nocw')
        name.append(f"c{self.ldamc}")
        name.append(f"g{self.ldamg}")
        name.append('mulcg' if self.ldam_mul_c_g else 'subcg')
        
        return '_'.join(name)

    def forward(self, x, target, group, targetgroup, instance_weights):
        if self.use_instance:
            assert instance_weights is not None
        index = torch.zeros_like(x, dtype=torch.uint8).to(device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))

        #do the same for groups
        gindex = torch.zeros(x.shape[0], self.g_list.shape[0],  dtype=torch.uint8).to(device)
        gindex.scatter_(1, group.data.view(-1, 1), 1)
        gindex_float = gindex.type(torch.FloatTensor).to(device)
        #the following line makes batch_g a vector
        batch_g = torch.matmul(self.g_list[None, :], gindex_float.transpose(0,1))
        batch_g = batch_g.view((-1, 1))

        #do the same for targetgroup combination
        mgindex = torch.zeros(x.shape[0], self.mg_list.shape[0],  dtype=torch.uint8).to(device)
        mgindex.scatter_(1, targetgroup.data.view(-1, 1), 1)
        mgindex_float = mgindex.type(torch.FloatTensor).to(device)
        #the following line makes batch_cg a vector
        batch_mg = torch.matmul(self.mg_list[None, :], mgindex_float.transpose(0,1))
        batch_mg = batch_mg.view((-1, 1))



        if self.ldam_mul_c_g:
            x_m = x - batch_m * batch_g
        else:
            x_m = x - self.ldamc * batch_m - self.ldamg * batch_g - self.ldamcg * batch_mg
    
        output = torch.where(index, x_m, x)
        regloss = 0
        if self.rho:
            for tval in range(self.m_list.shape[0]):
                reg = fair_reg(F.softmax(x[target==tval], dim=1)[:,tval], group[target==tval])
                regloss += reg

        if self.use_instance:
            return self.ce_instance(self.ldams * output, target, instance_weights=instance_weights) + self.rho * regloss
        else:
            return F.cross_entropy(self.ldams * output, target, weight=self.class_weight) + self.rho * regloss




class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    code from https://github.com/fursovia/self-adj-dice
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")

class CrossEntropyWithInstanceWeights(nn.Module):
    #supports both class weight and instance weight
    def __init__(self, class_weights=None, reduction='mean'):
        #don't reduce
        super(CrossEntropyWithInstanceWeights, self).__init__()
        self.celoss = nn.CrossEntropyLoss(weight=class_weights, reduce=False, reduction='none')
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, instance_weights: torch.Tensor) -> torch.Tensor:
        corss_ent_loss = self.celoss(logits, targets)
        if instance_weights is not None:
            assert corss_ent_loss.shape == instance_weights.shape, f"{corss_ent_loss.shape} != {instance_weights.shape}"
            corss_ent_loss = corss_ent_loss * instance_weights
        #reduce here
        if self.reduction == "mean":
            return corss_ent_loss.mean()
        elif self.reduction == "sum":
            return corss_ent_loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return corss_ent_loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")



if __name__ == '__main__':
    from imblearn.datasets import fetch_datasets
    from sklearn.preprocessing import LabelEncoder, StandardScaler 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import pdb

    le = LabelEncoder()
    dsname = 'abalone'
    imb_datasets = fetch_datasets(filter_data=[dsname])
    y = le.fit_transform(imb_datasets['abalone'].target)
    X = imb_datasets['abalone'].data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=0)
    
    X_train_tensor, X_val_tensor, X_test_tensor = torch.FloatTensor(X_train).to(device), torch.FloatTensor(X_val).to(device), torch.FloatTensor(X_test).to(device)
    y_train_tensor, y_val_tensor, y_test_tensor = torch.LongTensor(y_train).to(device), torch.LongTensor(y_val).to(device), torch.LongTensor(y_test).to(device)


    class MLP(nn.Module):
        def __init__(self, input_size, output_size, normed_linear=False):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, output_size)
            self.fc2 = NormedLinear(output_size, output_size) if normed_linear else nn.Linear(output_size, output_size) 
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    unique, counts = np.unique(y_train, return_counts=True)
    cls_num_list = counts.tolist()
    
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    criterion1 = FocalLoss(weight=per_cls_weights, gamma=1).to(device)
    criterion2 = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=30).to(device)
    criterion3 = SelfAdjDiceLoss().to(device)
    criterion4 = F.cross_entropy
    criterion5 = CrossEntropyWithInstanceWeights().to(device)
    
    criterions = {'ldam':criterion2, 'focal':criterion1, 'adjusteddice':criterion3, 'crossentropy':criterion4, 'cewithinstanceweights':criterion5}

    def f1_eval(model, x_tensor, y_true):
        model.eval()
        output = model(x_tensor)
        y_pred = F.softmax(output, dim=1).max(dim=1)[1].detach().cpu().numpy()
        return f1_score(y_true, y_pred, average='macro')
    
    for c_name, criterion in criterions.items():
        model = MLP(input_size=X.shape[1], output_size=np.max(y) + 1, normed_linear=True if c_name == 'ldam' else False).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        best_f1 = -1
        best_state_dict = model.state_dict()
        instance_weights = torch.ones(y_train_tensor.shape[0]).to(device)
        for i in range(1000):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_tensor)
            if c_name == 'cewithinstanceweights':
                #equal weight for all instances
                loss = criterion(output, y_train_tensor, instance_weights=instance_weights)        
            else:
                loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            val_f1 = f1_eval(model, X_val_tensor, y_val)
            if val_f1 > best_f1:
                best_state_dict = model.state_dict()
                best_f1 = val_f1
        model.load_state_dict(best_state_dict)
        test_f1 = f1_eval(model, X_test_tensor, y_test)
        print(c_name, test_f1)