import torch
import numpy as np
from torch.autograd import Variable




def deltaE(input, target):

	input  = input.view(input.shape[1], input.shape[2], input.shape[3])
    input  = input.data.numpy()
    target = target.view(target.shape[1], target.shape[2], target.shape[3])
    target = target.data.numpy() 

    # Converse YCbCr to RGB 
    # denormalization 
    input  = input  * 512 + 512
    target = target * 512 + 512

    # reshape to (360*480) x 3 => 172800 x 3
    input  = np.reshape(input, (-1, 3))
    target = np.reshape(target, (-1, 3))
    
    # Inverse Quantization 
    input  = inverseQuant(input, 10)
    target = inverseQuant(target, 10)

    # Clip 
    input  = clipImg(input)
    target = clipImg(target)

    # Y'CbCr to R'G'B' 
    R_prime_input = clipRGB(input[:, 0] + 1.47460 * input[:, 2])
    G_prime_input = clipRGB(input[:, 0] - 0.16455 * input[:, 1] - 0.57135 * input[:, 2])
    B_prime_input = clipRGB(input[:, 0] + 1.88140 * input[:, 1])

    R_prime_target = clipRGB(target[:, 0] + 1.47460 * target[:, 2])
    G_prime_target = clipRGB(target[:, 0] - 0.16455 * target[:, 1] - 0.57135 * target[:, 2])
    B_prime_target = clipRGB(target[:, 0] + 1.88140 * target[:, 1])

    # R'G'B' to RGB, and RGB normalization
    R_input  = inversePQ_TF(R_prime_input)
    G_input  = inversePQ_TF(G_prime_input)
    B_input  = inversePQ_TF(B_prime_input)

    R_target = inversePQ_TF(R_prime_target)
    G_target = inversePQ_TF(G_prime_target)
    B_target = inversePQ_TF(B_prime_target)