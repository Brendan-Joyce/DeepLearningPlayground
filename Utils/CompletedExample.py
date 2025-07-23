import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

######################
### LOSS FUNCTIONS ###
######################

class CustomLossFunction:
    def __init__(self, loss_lambda):
        self.Lambda = loss_lambda
        
    def __call__(self, *args: Any) -> Any:
        return self._calculate_loss(args[0]) * self.Lambda
    
    def _calculate_loss(self, _: nn.Module):
        raise NotImplementedError('Children must implement')
    
class Lasso(CustomLossFunction):
    def __init__(self, loss_lambda):
        super().__init__(loss_lambda)
        
    def _calculate_loss(self, model: nn.Module):
        return sum(p.abs().sum() for p in model.parameters())
        
class WeightDecay(CustomLossFunction):
    def __init__(self, loss_lambda):
        super().__init__(loss_lambda)
        
    def _calculate_loss(self, model: nn.Module):
        return sum(p.pow(2).sum() for p in model.parameters())


##################
### GLM MODELS ###
##################

class GlmModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def GetFeatureWeights():
        raise NotImplementedError('To be implemented by subclass')

class LinearRegressionModel(GlmModel):
    def __init__(self, numFeatures, learning_rate = .001, regularizations = []):
        super().__init__()
        self.Linear = nn.Linear(numFeatures, 1)
        self.Loss = torch.nn.MSELoss()
        self.Optim = optim.SGD(self.parameters(), lr = learning_rate)
        self.Regularization = regularizations

    def forward(self, x):
        return self.Linear(x)
    
    def train(self, x, y, num_epochs = 10):
        self.ReportedLoss = {}
        for epoch in range(num_epochs):
            y_pred = self.forward(x)
            reg_vals = sum([reg(self) for reg in self.Regularization])
            loss = self.Loss(y_pred, y) + reg_vals
            
            self.Optim.zero_grad()
            loss.backward()            
            self.Optim.step()
            
            self.ReportedLoss[epoch] = loss.item()

    def GetFeatureWeights(self):
        return self.Linear.weight, self.Linear.bias

class LogisticRegressionModel(GlmModel):
    def __init__(self, numFeatures, learning_rate = .001, regularizations = []):
        super().__init__()
        self.Linear = nn.Linear(numFeatures, 1)
        self.Loss = torch.nn.BCELoss()
        self.Optim = optim.SGD(self.parameters(), lr = learning_rate)
        self.Regularization = regularizations

    def forward(self, x):
        return torch.sigmoid(self.Linear(x))
    
    def train(self, x, y, num_epochs = 10):
        self.ReportedLoss = {}
        for epoch in range(num_epochs):
            y_pred = self.forward(x)
            reg_vals = sum([reg(self) for reg in self.Regularization])
            loss = self.Loss(y_pred, y) + reg_vals
            
            self.Optim.zero_grad()
            loss.backward()            
            self.Optim.step()
            
            self.ReportedLoss[epoch] = loss.item()

    def GetFeatureWeights(self):
        return self.Linear.weight, self.Linear.bias

class GlmModelWraper:
    def __init__(self, model, x_df: pd.DataFrame, y_df: pd.DataFrame, regularizations = [], learning_rate = .001):
        self.Model = model(x_df.shape[1], learning_rate = learning_rate, regularizations = regularizations)
        self.X = x_df
        self.Y = y_df

    def train_model(self,num_epochs = 100):
        x = torch.tensor(self.X.values).float()
        y = torch.tensor(self.Y.values).float()
        
        self.Model.train(x, y, num_epochs)
        
    def GetFeatureWeights(self):
        weights, bias = self.Model.GetFeatureWeights()
        param_names = self.X.columns.to_list() + ['bias']
        param_vals = weights.flatten().tolist() + bias.flatten().tolist()
        return list(zip(param_names, param_vals))
    
    def PlotLossOverTime(self):
        plt.plot(list(self.Model.ReportedLoss.values())[1:])
        plt.ylabel('MSE Loss')
        plt.xlabel('Epoch')
        plt.title('Loss by Epoch')
        plt.show()