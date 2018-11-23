import cv2
import torch
import numpy as np
from torch.autograd import Variable
from ptsemseg.loss.chromaUpSampling import chromaUpSampling 


def inverseQuant(Img, BitDepth):
    Img[:, 0] = (Img[: ,0] / (2 << (BitDepth-8)) -  16) / 219
    Img[:, 1] = (Img[: ,1] / (2 << (BitDepth-8)) - 128) / 224
    Img[:, 2] = (Img[: ,2] / (2 << (BitDepth-8)) - 128) / 224
    return Img

def clipRGB(x):
    return np.clip(x, 0, 1)

def clipImg(Img):
    Img[:, 0] = np.clip(Img[:, 0],   0,    1)
    Img[:, 1] = np.clip(Img[:, 1], -0.5, 0.5)
    Img[:, 2] = np.clip(Img[:, 2], -0.5, 0.5)
    return Img

def PQ_TF(x): 
    m1 = 2610.0/(4096*4)
    m2 = 2523.0*128/4096
    c1 = 3424.0/4096
    c2 = 2413.0*32/4096
    c3 = 2392.0*32/4096 
    x = x**m1
    return ((c1 + c2 * x) / (1 + c3 * x))**m2

def inversePQ_TF(x):
    m1 = 2610.0/(4096*4)
    m2 = 2523.0*128/4096
    c1 = 3424.0/4096
    c2 = 2413.0*32/4096
    c3 = 2392.0*32/4096
    x = x**(1/m2)
    numerator = np.max(x- c1, 0)
    denominator = c2 - c3 * x
    return (numerator/denominator)**(1/m1)

def PhilipsTF(x, y):
    rho = 25
    gamma = 2.4
    r = y / 5000.0
    N = np.log(1 + (rho - 1) * ((r*x)**(1/gamma)))
    M = np.log(1 + (rho - 1) * (r**(1/gamma))) 
    return N/M

def transfer(X, Y, Z):
    return (PQ_TF(X) + PhilipsTF(X, 10000) / 2, 
            PQ_TF(Y) + PhilipsTF(Y, 10000) / 2,
            PQ_TF(Z) + PhilipsTF(Z, 10000) / 2)

def RGB2XYZ(R, G, B):
    X = 0.636958 * R + 0.144617 * G + 0.168881 * B
    Y = 0.262700 * R + 0.677998 * G + 0.059302 * B
    Z = 0.000000 * R + 0.028073 * G + 1.060985 * B
    return (X, Y, Z)

def tPSNR(input, target):

    input  = input.view(input.shape[1], input.shape[2], input.shape[3])
    input  = input.data.numpy()
    target = target.view(target.shape[1], target.shape[2], target.shape[3])
    target = target.data.numpy()

    # denormalization
    input  = input * 512 + 512
    target = target *  512 + 512

    # reshape to (360*480) x 3
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

    # RGB to XYZ
    (X_input,  Y_input,  Z_input)  = RGB2XYZ(R_input,  G_input,  B_input)
    (X_target, Y_target, Z_target) = RGB2XYZ(R_target, G_target, B_target)
    
    # transfer function
    (X_prime_input,  Y_prime_input,  Z_prime_input)  = transfer(X_input,  Y_input,  Z_input) 
    (X_prime_target, Y_prime_target, Z_prime_target) = transfer(X_target, Y_target, Z_target)

    # reshape
    

    loss = Variable(torch.Tensor(np.array(loss)), requires_grad=True)

    return loss