import cv2
import torch
import numpy as np
from torch.autograd import Variable
from ptsemseg.loss.chromaUpSampling import chromaUpSampling 


def tPSNR(input, target):

    input = input.view(input.shape[1], input.shape[2], input.shape[3])
    input = input.data.numpy()
    target = target.view(target.shape[1], target.shape[2], target.shape[3])
    target = target.data.numpy()
    
    print (input.shape, target.shape) # (3x360x480, 3x360x480)
    for i in range(len(input[0])):
        temp1 = []
        temp2 = []
        for j in range(1,4):
            temp1.append(np.unit16(input[j][i]))
            temp2.append(np.unit16(target[j][i]))
        input_ycbcr.append(temp1)
        target_ycbcr.append(temp2)

    input_rgb = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2XYZ) for x in input_ycbcr]
    input_xyz = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2XYZ) for x in input_rgb]

    target_rgb = [cv2.cvtColor(np.array([[x]]), cv2.COLOR_YCrCb2RGB ) for x in target_ycbcr]
    target_xyz = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2XYZ) for x in target_rgb]
    
    mse = (np.square(np.array(input_xyz) - np.array(target_xyz))).mean(axis=0)
    loss = 10*np.log(10*len(input_ycbcr)/np.sum(mse))
    ### need modification

    loss = Variable(torch.Tensor(loss), requires_grad=True)

    return loss
