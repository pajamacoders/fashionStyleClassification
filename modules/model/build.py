from .model import ASEN
from .classification import StyleClassification

def build_model(cfg):
	return StyleClassification(cfg)