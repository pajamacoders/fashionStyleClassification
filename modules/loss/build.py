import torch
import torch.nn as nn


def build_loss(cfg):
	return {'triplet': TripletRankingLoss(cfg), 'celoss': ClsLoss()}


class TripletRankingLoss(nn.Module):
	def __init__(self, cfg):
		super(TripletRankingLoss, self).__init__()
		self.margin = cfg.SOLVER.MARGIN
		self.device = torch.device(cfg.DEVICE)
		self.criterion = nn.MarginRankingLoss(margin=self.margin)

	def forward(self, ref, pos, neg):
		x1 = nn.functional.cosine_similarity(ref, pos, dim=1)
		x2 = nn.functional.cosine_similarity(ref, neg, dim=1)
		target = torch.FloatTensor(ref.size(0)).fill_(1)
		target = target.to(self.device)
		loss = self.criterion(x1, x2, target)

		return loss

class ClsLoss(nn.Module):
	def __init__(self):
		super(ClsLoss, self).__init__()
		self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
	def forward(self, input, target):
		return self.cls_loss(input, target)