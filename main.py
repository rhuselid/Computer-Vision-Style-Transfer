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

def loss_fn():
	'''
	content and style losses
	'''
	pass

def optimization():
	'''
	optimizing between content loss (content image) and style loss (style image)
	'''
	pass

def gather_features(style=False):
	'''
	grabs the features from the vgg19 model and turns them into usable features
	 for this task

	style (bool): True if this is being used to extract the style features 
				(defaults to content features)
	'''
	pass

	
