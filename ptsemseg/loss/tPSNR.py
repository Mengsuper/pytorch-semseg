import torch
import numpy as np
from torch.autograd import Variable

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
    numerator = np.maximum(x- c1, 0)
    denominator = c2 - c3 * x
    return (numerator/denominator)**(1/m1)

def PhilipsTF(x, y):
    rho = 25
    gamma = 2.4
    r = y / 5000.0
    N = np.log(1 + (rho - 1) * ((r*x)**(1/gamma))) # log_e 
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

def SSE(X_input, X_target):
    return np.square(X_input - X_target).sum() 

def tPSNR(input, target):

    input  = input.view(input.shape[1], input.shape[2], input.shape[3])
    input  = input.data.numpy()
    target = target.view(target.shape[1], target.shape[2], target.shape[3])
    target = target.data.numpy()

    # denormalization 
    input  = input  * 1024 + 512
    target = target * 1024 + 512

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

    # RGB to XYZ
    (X_input,  Y_input,  Z_input)  = RGB2XYZ(R_input,  G_input,  B_input)
    (X_target, Y_target, Z_target) = RGB2XYZ(R_target, G_target, B_target)
    
    # transfer function
    (X_input,  Y_input,  Z_input)  = transfer(X_input,  Y_input,  Z_input) 
    (X_target, Y_target, Z_target) = transfer(X_target, Y_target, Z_target)

    # Sum of Square Error 
    SSEs = (SSE(X_input, X_target) + SSE(Y_input, Y_target) + SSE(Z_input, Z_target)) / 3
    
    # tPSNR
    loss = 10 * np.log10(1023 / SSEs)

    loss = Variable(torch.Tensor(np.array(loss)), requires_grad=True)

    return loss


def convToLab(x):
    if x.all() >= 0.008856:
        return x ** (1/3)
    else:
        return 7.78704 * x + 0.137931 

def XYZtoLab(X,Y,Z):
    Yn = 100
    invYn = 1.0 / Yn
    invXn = invYn / 0.95047
    invZn = invYn / 1.08883

    ylab = convToLab(Y * invYn)

    L = 116.0 * ylab - 16.0
    a = 500.0 * (convToLab(X * invXn) - ylab)
    b = 200.0 * (ylab - convToLab(Z * invZn))

    return (L, a, b)

def distanceDE(L1, a1, b1, L2, a2, b2):
# distance between two sample (L1,a1,b1) and (L2,a2,b2)

    DEG275 = 4.7996554429844
    DEG30 = 0.523598775598299
    DEG6 = 0.1047197551196598
    DEG63 = 1.099557428756428
    DEG25 = 0.436332

    cRef = np.sqrt(a1*a1 + b1*b1)
    cIn = np.sqrt(a2*a2 + b2*b2)

    cm = (cRef + cIn) / 2.0
    g = 0.5 * (1.0 - np.sqrt(cm ** 7.0 / (cm ** 7.0 + 25 ** 7.0)))

    aPRef = (1.0 + g) * a1
    aPIn = (1.0 + g) * a2

    cPRef = np.sqrt(aPRef * aPRef + b1 * b1)
    cPIn = np.sqrt(aPIn * aPIn + b2 * b2)

    hPRef = np.arctan(b1, aPRef)
    hPIn = np.arctan(b2, aPIn)

    deltaLp = L1 - L2
    deltaCp = cPRef - cPIn
    deltaHp = 2.0 * np.sqrt(cPRef * cPIn) * np.sin((hPRef - hPIn) / 2.0)

    lpm = (L1 + L2) / 2.0
    cpm = (cPRef + cPIn) / 2.0
    hpm = (hPRef + hPIn) / 2.0

    rC = 2.0 * np.sqrt( cpm ** 7.0 / ( cpm ** 7.0 + 25 ** 7.0 ) )
    dTheta = DEG30 * np.exp(-((hpm - DEG275) / DEG25) * ((hpm - DEG275) / DEG25))
    rT = - np.sin( 2.0 * dTheta ) * rC
    t = 1.0 - 0.17 * np.cos(hpm - DEG30) + 0.24 * np.cos(2.0 * hpm) + 0.32 * np.cos(3.0 * hpm + DEG6) - 0.20 * np.cos(4.0 * hpm - DEG63)

    sH = 1.0 + ( 0.015 * cpm * t )
    sC = 1.0 + ( 0.045 * cpm )
    sL = 1.0 + ( 0.015 * (lpm-50) * (lpm-50) / np.sqrt(20 + (lpm-50) * (lpm-50)) )

    deltaLpSL = deltaLp / sL
    deltaCpSC = deltaCp / sC
    deltaHpSH = deltaHp / sH
  
    DE = np.sqrt(deltaLpSL * deltaLpSL + deltaCpSC * deltaCpSC + deltaHpSH * deltaHpSH + rT * deltaCpSC * deltaHpSH )

    return DE


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

    # RGB to XYZ
    (X_input,  Y_input,  Z_input)  = RGB2XYZ(R_input,  G_input,  B_input)
    (X_target, Y_target, Z_target) = RGB2XYZ(R_target, G_target, B_target)

    # XYZ to Lab Space
    (L_input,  a_input,  b_input)  = XYZtoLab(X_input,  Y_input,  Z_input)
    (L_target, a_target, b_target) = XYZtoLab(X_target, Y_target, Z_target)

    # deltaE2000 distance DE
    DE = distanceDE(L_input,  a_input,  b_input, L_target, a_target, b_target)



    # PSNR_DE
    loss = 10 * np.log10(10000 / np.mean(DE))


    loss = Variable(torch.Tensor(np.array(loss)), requires_grad=True)

    return loss
