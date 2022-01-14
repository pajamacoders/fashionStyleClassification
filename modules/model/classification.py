import torch
import torch.nn as nn
from .model import ASEN

class StyleClassification(ASEN):
    def __init__(self, cfg):
        super(StyleClassification, self).__init__(cfg)
        self.fc = nn.Sequential(nn.Linear(1024,512), nn.Linear(512,3))

    def forward(self, *input, **kwargs):
        out, attmap = super().forward( *input, **kwargs)
        clsscore = self.fc(out)
        return out, attmap, clsscore

    def load_state_dict(self, checkpoint):
        state = super(StyleClassification, self).state_dict()
        for k in checkpoint:
            if k in state:
                state[k] = checkpoint[k]
        super(StyleClassification, self).load_state_dict(state)

            