# -*- coding: utf-8 -*-
# Author: yongyuan.name

#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os

import os
import h5py
import numpy as np
from numpy import linalg as LA


def get_imlist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

data_transforms = transforms.Compose([
                transforms.Scale(256),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406,], [0.229, 0.224, 0.225])
                ])

data_dir = '/Users/fanq15/Documents/pytorch/image-retrieval/database'
dsets = datasets.ImageFolder(data_dir, data_transforms)
print dsets.classes
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 1, shuffle = False, num_workers = 1)




model = torchvision.models.squeezenet1_0(pretrained = True)
model.train(False)
feats = []
names = []
for data in dset_loaders:
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    feat = model(inputs)
    
    img_name = labels
    norm_feat = feat.data.numpy()[0]/LA.norm(feat.data.numpy()[0])
    feats.append(norm_feat)
    names.append(img_name.data.numpy()[0])
    #print "image %d feature extraction, total %d images" %((i+1), len(img_list))
feats = np.array(feats)
h5f = h5py.File('featsCNN.h5', 'w')
h5f.create_dataset('dataset_1', data = feats)
h5f.create_dataset('dataset_2', data = names)
h5f.close()

