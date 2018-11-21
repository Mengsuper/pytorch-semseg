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

    input_ycbcr =[]
    target_ycbcr = []
    print (input.shape, target.shape) # (3x360x480, 3x360x480)

    for i in range(360):
        for j in range(480):
            temp1 = []
            temp2 = []
            for k in range(0,3):
                temp1.append(input[k][i][j])
                temp2.append(target[k][i][j])
            input_ycbcr.append(temp1)
            target_ycbcr.append(temp2)
    print(len(input_ycbcr))
    input_ycbcr = [[x[0],x[2], x[1]] for x in input_ycbcr]
    target_ycbcr = [[x[0],x[2], x[1]] for x in target_ycbcr]
    input_rgb = [cv2.cvtColor(np.array([[x]]), cv2.COLOR_YCrCb2RGB) for x in input_ycbcr]
    input_xyz = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2XYZ) for x in input_rgb]

    target_rgb = [cv2.cvtColor(np.array([[x]]), cv2.COLOR_YCrCb2RGB) for x in target_ycbcr]
    target_xyz = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2XYZ) for x in target_rgb]

    mse = (np.square(np.array(input_xyz) - np.array(target_xyz))).mean(axis=0)
    loss = 10*np.log(10*(2**10)/np.sum(mse))
    ### need modification

    loss = Variable(torch.Tensor(np.array(loss)), requires_grad=True)

    return loss
