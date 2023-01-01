import torch
import torch.nn as nn
import torch.nn.functional as F

class BadLinearTransformer(nn.Module):
    def __init__(self, in_width, neuro_num, normalization = None, activation = None, dropout = None):
        super(BadLinearTransformer, self).__init__()
        self.linearA = nn.Linear(in_width, neuro_num)
        self.linearB = nn.Linear(in_width, neuro_num)
        self.linearC = nn.Linear(in_width, neuro_num)
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout
    
    def forward(self, x):
        a = self.linearA(x)
        b = self.linearB(x)
        c = self.linearC(x)
        if self.activation is not None:
            a = self.activation(a)
            b = self.activation(b)
            c = self.activation(c)
        x = a * b + c
        if dropout is not None:
            x = self.dropout(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x
