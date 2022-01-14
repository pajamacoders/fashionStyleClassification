from .model import ASEN
from .classification import StyleClassification
from .arcmargin_classifier import ArcStyleClassification

def build_model(cfg):
	if cfg.MODEL.TYPE=='ArcStyleClassification':
		return ArcStyleClassification(cfg)
	elif cfg.MODEL.TYPE=='StyleClassification':
		return StyleClassification(cfg)