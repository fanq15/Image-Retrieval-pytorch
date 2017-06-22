# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from numpy import linalg as LA
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import h5py

data_transforms = transforms.Compose([
                transforms.Scale((224,224)),
                #transforms.CenterCrop(256),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406,], [0.229, 0.224, 0.225])
                ])

data_dir = '/Users/fanq15/Documents/pytorch/101_ObjectCategories'
dsets = datasets.ImageFolder(data_dir, data_transforms)
#print dsets.classes

dset_loaders = torch.utils.data.DataLoader(dsets, batch_size = 1, shuffle = False, num_workers = 4)

model = torchvision.models.resnet18(pretrained = True)
model.train(False)
#print model

class AlexNetConv4(nn.Module):
            def __init__(self):
                super(AlexNetConv4, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(model.features.children())[:-2]
                )
            def forward(self, x):
                x = self.features(x)
                return x
#new_model = AlexNetConv4()
#print new_model

class MyResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, transform_input=False):
        super(MyResNetFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        #self.fc = resnet.fc
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.conv1(x)
        # 149 x 149 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # 147 x 147 x 32
        x = self.layer1(x)
        # 147 x 147 x 64
        x = self.layer2(x)
        # 73 x 73 x 64
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=7)

        return x


my_resnet = MyResNetFeatureExtractor(model)
print my_resnet

feats = []
names = []
since = time.time()
for data in dset_loaders:
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    feat = my_resnet(inputs)
    feat = torch.sum(feat, dim = 2)
    feat = torch.sum(feat, dim = 3)
    feat = torch.squeeze(feat)
    
    img_name = labels
    norm_feat = feat.data.numpy()/LA.norm(feat.data.numpy())
    feats.append(norm_feat)
    names.append(img_name.data.numpy()[0])
    #print "image %d feature extraction, total %d images" %((i+1), len(img_list))
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
feats = np.array(feats)
h5f = h5py.File('featsCNN.h5', 'w')
h5f.create_dataset('dataset_1', data = feats)
h5f.create_dataset('dataset_2', data = names)
h5f.close()

