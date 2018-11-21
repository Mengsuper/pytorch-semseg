import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.loss.chromaUpSampling import chromaUpSampling
import ptsemseg.loss.pytorch_ssim
from torch.autograd import Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    print (input.size(), target.size())

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def chrom_downsampling_loss(input, target, weight=None, size_average=True):
    Luma = input[0]
    Chroma = input[1]
    ChromaA = Chroma[:, 0:1, :, :]
    ChromaB = Chroma[:, 1:2, :, :]
    #print (input[1].size(), Luma.size(), ChromaA.size(), ChromaB.size(), target.size())
    input = chromaUpSampling(Luma, ChromaA, ChromaB, '420', 'MPEG_CfE')
    MSELoss = nn.MSELoss() # dummy loss only for test
    #loss = MSELoss(input, target)
    loss = tPSNR(input, target)

    return loss
    
def ssim_loss_function(input, target, weight=None, size_average=True):
    
    #print(input[0].size())
    #print(input[1].size())
    Luma = input[0]
    Chroma = input[1]
    ChromaA = Chroma[:, 0:1, :, :]
    ChromaB = Chroma[:, 1:2, :, :]
    input = chromaUpSampling(Luma, ChromaA, ChromaB, '420', 'MPEG_CfE')
    #print(input.size())

    #img1_0 = np.reshape(input, (1,3,360,480))
    #img1_1 = torch.from_numpy(img1_0)
    img1_2 = Variable(input)

    #img2_0 = np.reshape(target, (1,3,360,480))
    #img2_1 = torch.from_numpy(img1_0)
    img2_2 = Variable(target)

    #pytorch_ssim.ssim(img1_2, img2_2) #calculating the ssim value for testing
    ssim_loss = ptsemseg.loss.pytorch_ssim.SSIM(window_size = 11)
    loss = Variable(ssim_loss(img1_2, img2_2), requires_grad = True)
    #ssim_loss = ptsemseg.loss.pytorch_ssim.ssim(img1_2, img2_2)
    #loss = Variable(ssim_loss, requires_grad = True)
    return loss


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target, 
                                  K, 
                                  weight=None, 
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, 
                                   target, 
                                   K, 
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, 
                               target, 
                               weight=weight, 
                               reduce=False,
                               size_average=False, 
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
