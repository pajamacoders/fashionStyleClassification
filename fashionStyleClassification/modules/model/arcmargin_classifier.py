from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from .model import ASEN

class ArcStyleClassification(nn.Module):
    def __init__(self, cfg):
        super(ArcStyleClassification, self).__init__()
        self.body = ASEN(cfg)
        self.head = ArcMarginProduct(cfg)
        self.fc = nn.Linear(2048,1024)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(2048,1024),
            nn.Linear(1024,1024),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(1024,512))

    def forward(self, x, a, level='global', label=None):
        out, attmap, greature = self.body.forward(x, a, level=level)
        
        greature=self.fc(greature.flatten(1))
        mixed_feature = self.fc1(torch.cat([out,greature], dim=-1))
        if label is not None:
            cls_score = self.head.forward_train(mixed_feature, label)
        return out, attmap, cls_score
    
    def forward_test(self, x, a, level='global'):
        out, attmap, greature = self.body.forward(x, a, level=level)
        greature=self.fc(greature.flatten(1))
        mixed_feature = self.fc1(torch.cat([out,greature], dim=-1))
        cls_score = self.head.simple_test(mixed_feature)
        return out, attmap, cls_score

    def load_state_dict(self, checkpoint):
        state = self.state_dict()
        for k in checkpoint:
            if k in state:
                state[k] = checkpoint[k]
        super(ArcStyleClassification, self).load_state_dict(state)



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, cfg):#in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = cfg.HEAD.in_features
        self.out_features = cfg.HEAD.out_features
        self.s = cfg.HEAD.s
        self.m = cfg.HEAD.m
        self.weight = Parameter(torch.FloatTensor(cfg.HEAD.out_features, cfg.HEAD.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.device = cfg.DEVICE
        
        self.easy_margin = cfg.HEAD.easy_margin
        self.cos_m = math.cos(cfg.HEAD.m)
        self.sin_m = math.sin(cfg.HEAD.m)
        self.th = math.cos(math.pi - cfg.HEAD.m)
        self.mm = math.sin(math.pi - cfg.HEAD.m) * cfg.HEAD.m

    def forward(self, input, label):
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output

    def forward_train(self, input, labels):
        if isinstance(input, (tuple,list)):
            input = input[-1]
        logits = self.forward(input, labels)
        return logits

    def simple_test(self, input):
        """Test without augmentation."""
        if isinstance(input, (tuple,list)):
            input = input[-1]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine
