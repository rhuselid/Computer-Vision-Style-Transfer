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

from PIL import Image
import matplotlib.pyplot as plt



vgg19 = models.vgg19(pretrained=True).float()
# pretrained object detection deep CNN to use for feature extraction
for param in vgg19.parameters():
    param.requires_grad = False

device = torch.device('cpu')


def total_loss(content_loss, style_loss, alpha=0.5, beta=0.5):
	return content_loss*alpha + style_loss*beta


def style_loss(style_img: torch.Tensor, noise: torch.Tensor): 
	"""
	returns the style loss between a style image layer and its corresponding
	noise layer
	"""
	# print(type(style_img), type(noise))
	# print(style_img[0])
	# print(noise[0])
	# print(type(style_img[0]))
	# print(type(noise[0]))

	# print(style_img[0].requires_grad)
	# print(noise[0].requires_grad)

	# print(style_img.shape)
	# print(noise.shape)
	total_style_loss = 0

	for i in range(len(style_img)):
		# this is a list of tensors

		total_style_loss += torch.sum(torch.pow(noise[i] - style_img[i], 2)).div(4*(np.power(noise[i].shape[0]*noise[i].shape[1], 2)))
		# origional:
		# style_gram = torch.mm(style_img[i], style_image[i].t())
		# noise_gram = torch.mm(noise, noise.t())
		# torch.sum(torch.pow(noise_gram - style_gram, 2)).div(4*(np.power(noise.shape[0]*noise.shape[1], 2)))

	return total_style_loss

def content_loss(content_img: torch.Tensor, noise: torch.Tensor):
	"""
	returns the content loss between the content image layer and corresponding
	noise layer
	"""
	return torch.sum(torch.pow(content_img - noise, 2)).div(2)


def create_optimizer(params, lr):
	'''
	optimizing between content loss (content image) and style loss (style image)
	'''
	return torch.optim.Adam(params=params, lr=lr)


def gather_features(x, style=False, noise=False):
	'''
	grabs the features from the vgg19 model and turns them into usable features
	 for this task


	x (tensor): image to model
	style (bool): True if this is being used to extract the style features 
				(defaults to content features)
	'''
	global vgg19

	intermediary_outputs = []
	style_layers = [4,7,12,21,30]
	# we want the output from these layers as our style features
	content_layers = [23]
	# we want the output from this chunk to be our content features

	max_depth = max(max(style_layers), max(content_layers))

	for i,layer in enumerate(vgg19.features):

		if i > max_depth:
			break

		x = layer(x)
		# making the forward passes of the model

		# note that the arrays are added as numpy arrays
		if style:
			if not noise:
				if i in style_layers:
					intermediary_outputs.append(x.detach())
			else:
				if i in style_layers:
					intermediary_outputs.append(x)
		
		else:
			# content
			if not noise:
				if i in content_layers:
					intermediary_outputs.append(x.detach())
				else:
					intermediary_outputs.append(x)

	# note that intermediary outputs is a list of numpy arrays of different shapes (differ between layers)

	if not style:

		return intermediary_outputs[0]

	else:
		# we have to create the cross correlation for the style matrix

		gram_style_features = []

		for intermediary_layer in intermediary_outputs:

			# unroll the layer output into column form (vectorized multiplication)
			current_shape = intermediary_layer.shape
			# print(current_shape)
			# .squeeze().view(intermediary_outputs[i].shape[1], -1)
			reshaped_layer = intermediary_layer.squeeze().view(current_shape[1], -1)

			gram_matrix = torch.mm(reshaped_layer, reshaped_layer.T)

			gram_style_features.append(gram_matrix)

		return gram_style_features


def generate_output(content_image, style_image):
	batch_num_c, content_rows, content_cols, content_channels = content_image.shape
	batch_num_s, style_rows, style_cols, style_channels = style_image.shape

	noise = torch.randn(1, content_rows, content_cols, content_channels, device=device, requires_grad=True)

	content_activation = gather_features(content_image, style=False)
	style_activations = gather_features(style_image, style=True)
	
	iterations = 250
	start_lr = 0.1

	for iter_num in range(iterations):
		current_learn_rate = max(1e-5, start_lr*(0.97**iter_num))

		optimizer = create_optimizer([noise], current_learn_rate)
		noise_content_activation = gather_features(noise, style=False)
		# noise_content_activation.requires_grad = True

		noise_style_activations = gather_features(noise, style=True)

		# for i in range(len(noise_style_activations)):
		# 	noise_style_activations[i].requires_grad = True

		optimizer.zero_grad()

		current_content_loss = content_loss(content_activation, noise_content_activation)
		current_style_loss = style_loss(style_activations, noise_style_activations)
		current_total_loss = total_loss(current_content_loss, current_style_loss, alpha=0.001, beta=0.999)

		current_total_loss.backward()
		optimizer.step()

		if iter_num % 25 == 0:
			print('Iteration Number', iter_num)
			print('Style Loss:   ', int(current_style_loss.item()))
			print('Content Loss: ', int(current_content_loss.item()))
			print()

	return noise
	

if __name__ == '__main__':

	content_path = './data/content/Cat.jpeg'
	style_path = './data/style/VanGogh01.jpg'

	content_image = np.array(Image.open(content_path)) / 255
	content_image = np.moveaxis(content_image, -1, 0)
	content_image = torch.from_numpy(content_image).unsqueeze(0).float()

	style_image = np.array(Image.open(style_path)) / 255
	style_image = np.moveaxis(style_image, -1, 0)
	style_image = torch.from_numpy(style_image).unsqueeze(0).float()

	output = generate_output(content_image, style_image).detach().numpy()[0,:,:,:]
	output = np.moveaxis(output, 0, -1)

	plt.imshow(output)
	plt.show()

# gather_features(torch.rand((64, 3, 32, 32)), style=True)


