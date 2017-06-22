# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
import h5py

h5f = h5py.File('featsCNN.h5','r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

acc = 0
top_k = 5
item_num = len(imgNames)

for i in range(item_num):
    queryVec = feats[i]
    imgname = imgNames[i]
    scores = np.dot(queryVec, feats.T)

    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    #print rank_ID
    #print rank_score

    maxres = top_k
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    #print imlist
    i_total = 0
    for item in imlist:
        if item == imgname:
            i_total += 1
    i_acc = i_total / float(maxres)
    acc += i_acc

acc = acc / float(item_num)
print 'top %s accuracy is %s' % (top_k, acc)



