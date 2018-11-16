import torch
import numpy as np
import scipy.ndimage as ndimg
from torch.autograd import Variable

def chromaUpSampling(Luma, ChromaA, ChromaB, Sampling, Filter):

	# better way to reduce dimension?? (this method make copy of data)
	# Luma shape: 1x1x360x480, Chroma: 1x180x240
	Luma = Luma.view(Luma.shape[2], Luma.shape[3])
	ChromaA = ChromaA.view(ChromaA.shape[1], ChromaA.shape[2])
	ChromaB = ChromaB.view(ChromaB.shape[1], ChromaB.shape[2])

	# https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
	Luma = Luma[:, :, np.newaxis]
	ImgUp = np.tile(Luma.data.numpy(), (1, 1, 3))

	if Sampling == '420':
	    subSampleMat = torch.Tensor([[1, 1], [1, 1]])
	elif Sampling == '422':
	    subSampleMat = torch.Tensor([[1, 1]])
	elif Sampling == '444':
	    subSampleMat = torch.Tensor([1])   
	
	if Sampling == '420':
	    if 'MPEG' in Filter:
	        if 'CfE' in Filter:
	            D_0 = torch.Tensor([[-2, 16, 54, -4]])
	            D_1 = torch.Tensor([[-4, 54, 16, -2]])
	            C_0 = torch.Tensor([[1]])
	            C_1 = torch.Tensor([[-4, 36, 36, -4]])
	            offset_1 = 32
	            shift_1  = 6
	            shift_2  = 12
	            offset_2 = 2048
	        elif 'SuperAnchor' in Filter:
	            D_0 = torch.Tensor([[256]])
	            D_1 = torch.Tensor([-16, 144, 144, -16])
	            C_0 = torch.Tensor([256])
	            C_1 = torch.Tensor([[-16, 144, 144, -16]])
	            offset_1 = 32768
	            shift_1  = 16
	            shift_2  = 16
	            offset_2 = 32768
	            
	        for ChrIdx in range(1, 3):
	            if ChrIdx == 1:
	                s = ChromaA.data.numpy()
	            else:
	                s = ChromaB.data.numpy()
	            
	            f = np.zeros([s.shape[0]*2, s.shape[1]])
	            r = np.zeros([s.shape[0]*2, s.shape[1]*2])
	            # Horizontal filtering
	            f[0::2, :] = ndimg.correlate(s, D_0, mode = 'nearest')
	            f[1::2, :] = ndimg.correlate(s, D_1, mode = 'nearest')

	            if 'Float' in Filter:
	                r[:, 0::2] = (ndimg.correlate(f, C_0, mode = 'nearest') + offset_1) / 2^shift_1
	                r[:, 1::2] = (ndimg.correlate(f, C_1, mode = 'nearest') + offset_2) / 2^shift_2
	                ImgUp[:, :, ChrIdx]  = r
	            else:
	                r[:, 0::2] = np.double((np.uint32(ndimg.correlate(f, C_0, mode = 'nearest')+ offset_1)) >> shift_2)
	                r[:, 1::2] = np.double((np.uint32(ndimg.correlate(f, C_1, mode = 'nearest')+ offset_2)) >> shift_2)
	                ImgUp[:, :, ChrIdx]  = r
	    else:
	        ImgUp[:, :, 1] = np.kron(ChromaA, subSampleMat) 
	        ImgUp[:, :, 2] = np.kron(ChromaB, subSampleMat)
	    
	elif Sampling == '422':
	    ImgUp[:, :, 1] = np.kron(ChromaA, subSampleMat)
	    ImgUp[:, :, 2] = np.kron(ChromaB, subSampleMat)

	ImgUp = Variable(torch.Tensor(ImgUp.reshape(ImgUp.shape[2], ImgUp.shape[0], ImgUp.shape[1])), requires_grad=True)
	ImgUp = ImgUp.unsqueeze(0)
	return ImgUp
