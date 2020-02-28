import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models

import numpy as np
import os
import math


vgg19 = models.vgg19(pretrained=True)
# pretrained object detection deep CNN to use for feature extraction

def total_loss(content_loss, style_loss, alpha=0.5, beta=0.5):
	return content_loss*alpha + style_loss*beta


def style_loss(style_img: torch.Tensor, noise: torch.Tensor): 
	"""
	TODO: returns the style loss between a style image layer and its corresponding
	noise layer
	"""
	pass

def content_loss(content_img: torch.Tensor, noise: torch.Tensor):
	"""
	returns the content loss between the content image layer and corresponding
	noise layer
	"""
	return torch.sum(torch.pow(content_img - noise, 2)) / 2


def optimization():
	'''
	optimizing between content loss (content image) and style loss (style image)
	'''
	pass

def gather_features(x, style=False):
	'''
	grabs the features from the vgg19 model and turns them into usable features
	 for this task


	x (tensor): image to model
	style (bool): True if this is being used to extract the style features 
				(defaults to content features)
	'''
	intermediary_outputs = []
	style_layers = [4,7,12,21,30]
	# we want the output from these layers as our style features
	content_layers = list(range(23))
	# we want the output from this chunk to be our content features

	max_depth = max(max(style_layers), max(content_layers))

	for i,layer in enumerate(vgg19.features):

		if i > max_depth:
			break

		x = layer(x)
		# making the forward passes of the model

		# note that the arrays are added as numpy arrays
		if style:
			if i in style_layers:
				intermediary_outputs.append(x.detach().numpy())
		
		else:
			# content
			if i in content_layers:
				intermediary_outputs.append(x.detach().numpy())


	# note that intermediary outputs is a list of numpy arrays of different shapes (differ between layers)

	if not style:

		return intermediary_outputs

	else:
		# we have to create the cross correlation for the style matrix
		gram_style_features = []

		for intermediary_layer in intermediary_outputs:

			# unroll the layer output into column form (vectorized multiplication)
			current_shape = intermediary_layer.shape
			reshaped_layer = intermediary_layer.reshape(current_shape[1], -1)

			gram_matrix = np.matmul(reshaped_layer, reshaped_layer.T).shape

			gram_style_features.append(gram_matrix)

		return gram_style_features




gather_features(torch.rand((64, 3, 32, 32)), style=True)


