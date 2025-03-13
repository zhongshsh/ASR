import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .modules import *

class ASRBatchNorm2d(nn.Module):
    def __init__(self, num_features, module):
        super(ASRBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.fused_bn = nn.BatchNorm2d(num_features)
        
        self.vector = Parameter(torch.ones(1, num_features, 1, 1).cuda())
        nn.init.constant_(self.vector, 0.1)
        
        self.attention_module = eval(module)(planes=num_features)
        self.deploy = False
    
    def forward(self, x):
        if self.deploy:
            x = self.fused_bn(x)
        else:
            x = self.bn(x)
            attention_weights = self.attention_module(self.vector)
            x = x * attention_weights
        return x

    def switch_to_deploy(self, delete=False):
        self.deploy = True
        gamma = self.bn.weight
        beta = self.bn.bias
        attention_weights = self.attention_module(self.vector).reshape(-1)
        
        gamma_fused = gamma * attention_weights
        beta_fused = beta * attention_weights
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        
        self.fused_bn.weight.data = gamma_fused
        self.fused_bn.bias.data = beta_fused
        self.fused_bn.running_mean = running_mean
        self.fused_bn.running_var = running_var
        self.fused_bn.eval()
        
        # If the model is only used for inference, we can remove unnecessary modules to accelerate the inference process.
        if delete:
            del self.bn
            del self.vector
            del self.attention_module
        
    def switch_to_train(self):
        self.deploy = False
        
        