import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class EBGan:
	def __init__(self, lambda_pt, margin):
		self.lambda_pt = lambda_pt
		self.margin = margin
		self.loss = nn.MSELoss()

	def pullaway_loss(embeddings):
		norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
		normalized_emb = embeddings / norm
		similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
		batch_size = embeddings.size(0)
		loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
		return loss_pt

	def gloss(recon_imgs, gen_imgs, img_embeddings):
		g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + self.lambda_pt * pullaway_loss(img_embeddings)
		return g_loss


	def dloss(real_recon, real_imgs, fake_recon, gen_imgs):
		d_loss_real = pixelwise_loss(real_recon, real_imgs)
		d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

		d_loss = d_loss_real
		if (self.margin - d_loss_fake.data).item() > 0:
			d_loss += self.margin - d_loss_fake

		return d_loss


