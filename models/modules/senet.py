"""Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
"""
import torch.nn as nn


__all__ = ["senet"]

class Attention(nn.Module):
    def __init__(self, planes):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, out):
        out = self.GlobalAvg(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        out = out.view(out.size(0), out.size(1), 1, 1)
        
        return out
        
def senet(**kwargs):
    return Attention(**kwargs)


