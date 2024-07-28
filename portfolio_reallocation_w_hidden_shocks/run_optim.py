import torch.optim as optim
import torch
dtype = torch.float
device = torch.device("cuda:0") # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
from run_nn import model
import torch.nn as nn
import numpy as np
import config
from payoff_sim import BS_payoff_sim
import os
import sys
sys.path.append(os.getcwd())
set_seed = 0
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def weighted_MSELoss(y_pred, y_tru, weight):
    return (((weight*(y_pred - y_tru))**2).sum(axis=1)).mean()

class run_optim:
    def __init__(self, X_y_set, J_set, miu_sigma_set, n_param, portfolio_size, X_init, rf):
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.2)
        self.n_param = n_param
        self.portfolio_size = portfolio_size
        self.X_init = X_init
        self.X_train = X_y_set[0]
        self.X_test = X_y_set[1]
        self.y_train = X_y_set[2]
        self.y_test = X_y_set[3]
        self.J_train = J_set[0]
        self.J_test = J_set[1]
        self.miu_train = miu_sigma_set[0]
        self.miu_test = miu_sigma_set[1]
        self.sigma_train = miu_sigma_set[2]
        self.sigma_test = miu_sigma_set[3]
        self.rf = rf
        self.batch_size = round(len(self.X_train)/2)

    @staticmethod
    def loss_function(x):
        if x == 'MSE':
            return nn.MSELoss()
        elif x == 'cross entropy':
            return nn.CrossEntropyLoss()

    def model_train(self):
        optimizer = self.optimizer
        X_train = self.X_train
        y_train = self.y_train
        J_train = self.J_train
        miu_train = self.miu_train
        sigma_train = self.sigma_train
        iter = 1000

        for i in range(iter):
            optimizer.zero_grad()
            y_pred = model(X_train)
            # loss = run_optim.loss_function('MSE')(model(X_train[i:i+batch_size]), y_train[i:i+batch_size])
            # loss = weighted_MSELoss(y_pred, y_train, torch.tensor([5, 1]))
            if i == 0:
                print('The initial loss:')
                loss = weighted_MSELoss(y_pred, y_train, torch.tensor([5, 1]))
                print(loss)
                # Uncomment to check if param are updating with small # of layers and samples
                # print('The initial bias and weight:')
                # print(model.weight())
                # print('The initial linear param:')
                # print(model.state_dict()['m.weight'][0])
                # print(model.state_dict()['m.bias'][0])
                # print('The initial convd param:')
                # print(model.state_dict()['conv.weight'][0])
                # print(model.state_dict()['conv.bias'][0])
            
            else:
                bias = model.weight()[0]
                weight = model.weight()[1]
                y_adj = torch.Tensor([])
                unit_weight = []
                ret_vol = []
                for j in range(len(miu_train)):
                    _weight = weight[j][0].tolist()
                    _bias = bias[j].item()
                    _weight.append(_bias/self.rf)
                    _min_max_weight = [(x-np.min(_weight))/(np.max(_weight)-np.min(_weight)) for x in _weight]
                    _unit_weight = [x/sum(_min_max_weight) for x in _min_max_weight[:-1]]
                    unit_weight.append(_unit_weight)
                    J_shock = [J_train[j][0][0].detach().numpy(), J_train[j][0][1].detach().numpy()]
                    obj = BS_payoff_sim(miu = miu_train[j], sigma = sigma_train[j], rho = -0.2, J_apt = J_shock, weight = np.array(_unit_weight), _lambda = 5, S_0 = 400, T = 1, t = 0, time_step = config.time_step, N = config.N, shock = True)
                    path = obj.exact().mean(axis = 0)
                    ret = np.prod(1+np.diff(path)/path[:-1])-1
                    vol = np.std(np.diff(path)/path[:-1])*config.time_step**0.5
                    ret_vol.append([ret, vol])
                    sr_shock = ret/vol
                    sr = (y_train[j][0]/y_train[j][1]).item()
                    y_adj = torch.cat((y_adj, torch.Tensor([1, 0]).view(1, 2) if sr_shock >= sr else torch.Tensor([0, 1]).view(1, 2)), 0)
                loss = weighted_MSELoss(y_pred, y_adj, torch.tensor([5, 1]))
            loss.backward()
            optimizer.step()

        print("Loss for training set is")
        print(loss)
        print('Final Asset Allocation is')
        print(unit_weight)
        print('Original Return and Vol:')
        print(y_train)
        print('Final Return and Vol:')
        print(ret_vol)

        # Uncomment to check if param are updating with small # of layers and samples
        # print('The final bias and weight:')
        # print(model.weight())
        # print('The final linear param:')
        # print(model.state_dict()['m.weight'][0])
        # print(model.state_dict()['m.bias'][0])
        # print('The final convd param:')
        # print(model.state_dict()['conv.weight'][0])
        # print(model.state_dict()['conv.bias'][0])

    def model_test(self): 
        run_optim.model_train(self)  

        #Placeholder for Testing

if __name__ == "__main__":
    import itertools
    portfolio_size = config.portfolio_size
    n_param = config.n_param
    X_init = np.array([config.miu_asset, config.sigma_1_asset, config.sigma_2_asset])
    J_init = np.array([config.J_apt_1_asset, config.J_apt_2_asset])
    X = torch.Tensor([])
    J = torch.Tensor([])
    y = torch.Tensor([])
    miu = []
    sigma = []
    for s in itertools.combinations(np.transpose(X_init), portfolio_size):
        obj_in = np.transpose(s)
        X = torch.cat((X, torch.Tensor(obj_in).view(-1, 1, n_param-1, portfolio_size)), 0)
        _miu = np.array(obj_in[0])
        _sigma = [np.array(obj_in[1]), np.array(obj_in[2])]
        miu.append(_miu)
        sigma.append(_sigma)
        weight = np.array([1/len(_miu)]*len(_miu))
        obj = BS_payoff_sim(miu = _miu, sigma = _sigma, rho = -0.2, J_apt = 'NA', weight = weight, _lambda = 5, S_0 = 400, T = 1, t = 0, time_step = config.time_step, N = config.N, shock = False)
        path = obj.exact().mean(axis = 0)
        ret = np.prod(1+np.diff(path)/path[:-1])-1
        vol = np.std(np.diff(path)/path[:-1])*config.time_step**0.5
        y = torch.cat((y, torch.Tensor([ret, vol]).view(1, 2)), 0)
    X.requires_grad = True
    for s in itertools.combinations(np.transpose(J_init), config.portfolio_size):
        J = torch.cat((J, torch.Tensor(np.transpose(s)).view(-1, 1, 2, portfolio_size)), 0)
    train_split = round(len(X) * 1)
    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]
    J_train, J_test = J[:train_split], J[train_split:]
    miu_train, miu_test = miu[:train_split], miu[train_split:]
    sigma_train, sigma_test = sigma[:train_split], sigma[train_split:]
    X_y_set = X_train, X_test, y_train, y_test
    J_set = J_train, J_test
    miu_sigma_set = miu_train, miu_test, sigma_train, sigma_test
    
    run_optim(X_y_set, J_set, miu_sigma_set, n_param, portfolio_size, X_init, rf = config.risk_free).model_test()
