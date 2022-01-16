import os
from .modules.config import cfg
from .modules.model import build_model
from .modules.data.transforms import GlobalTransform
import torch
import json
from PIL import Image

cfg_files ={'Kfashion':'config/Kfashion/KFashion.yaml',
            'Fashion14k':'config/Fashion14k/Fashion14k.yaml'}
class FashionStyleClassification:
    def __init__(self, cfg_path):
        assert isinstance(cfg_path,str), 'cfg_path must be string'
        with open(cfg_path,'r') as f:
            self.runconfig = json.load(f)
        
        model_cfg = self.runconfig.get('model_cfg',None)
        try:
            repopath = os.path.dirname(__file__)
            if model_cfg is not None:
                cfg.merge_from_file(os.path.join(repopath,cfg_files[model_cfg]))
        except FileNotFoundError as f:
            print(f'{os.path.join(repopath,cfg_files[model_cfg])} does not exist')
        cfg.freeze()
        self.device = torch.device(cfg.DEVICE)
        self.model = build_model(cfg)
        self.model.to(self.device)

        self.catid2style=self.runconfig['category']
        self.catid2style={int(k):v for k,v in self.catid2style.items()}
        # inference image transformation scheme
        self.gt = GlobalTransform(cfg)
        self.att_id = torch.tensor([0]).long().to(self.device)
        self.init_model(self.runconfig['pretrained'])
        self.model.eval()

    def init_model(self, pretrained=None):
        assert pretrained is not None, 'pretrained checkpoint path should be given'
        checkpoint = torch.load(pretrained, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
    
    def inference(self,x):
        """
        x: input image, numpy array, shape (h,w,c) channel type = BGR
        """
        inimg = Image.fromarray(x)
        inimg = self.gt(inimg)
        inimg = inimg.to(self.device).unsqueeze(0)
        with torch.no_grad():
            g_feats, attmap, cls_score = self.model.forward_test(inimg, self.att_id, level='global')
        cls_score=cls_score.squeeze()
        cls_score = cls_score.softmax(dim=-1)
        cls_id = cls_score.argmax(dim=-1)
        return {'style':self.catid2style[cls_id.item()], 'score':cls_score[cls_id].item()}
        
        

