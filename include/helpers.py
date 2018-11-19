import torch 
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

from torch.autograd import Variable





import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import PIL

from torch.autograd import Variable

def load_and_crop(imgname,target_width=512,target_height=512):
	'''
	imgname: string of image location
	load an image, and center-crop if the image is large enough, else return none
	'''
	img = Image.open(imgname)
	width, height = img.size
	if width <= target_width or height <= target_height:
		return None	
	
	left = (width - target_width)/2
	top = (height - target_height)/2
	right = (width + target_width)/2
	bottom = (height + target_height)/2
	
	return img.crop((left, top, right, bottom))

def save_np_img(img,filename):
    if(img.shape[0] == 1):
        plt.imshow(np.clip(img[0],0,1),cmap='Greys',interpolation='nearest')
    else:
        plt.imshow(np.clip(img.transpose(1, 2, 0),0,1))
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.
    
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def rgb2ycbcr(img):
    #out = color.rgb2ycbcr( img.transpose(1, 2, 0) )
    #return out.transpose(2,0,1)/256.
    r,g,b = img[0],img[1],img[2]
    y = 0.299*r+0.587*g+0.114*b
    cb = 0.5 - 0.168736*r - 0.331264*g + 0.5*b
    cr = 0.5 + 0.5*r - 0.418588*g - 0.081312*b
    return np.array([y,cb,cr])

def ycbcr2rgb(img):
    #out = color.ycbcr2rgb( 256.*img.transpose(1, 2, 0) )
    #return (out.transpose(2,0,1) - np.min(out))/(np.max(out)-np.min(out))
    y,cb,cr = img[0],img[1],img[2]
    r = y + 1.402*(cr-0.5)
    g = y - 0.344136*(cb-0.5) - 0.714136*(cr-0.5)
    b = y + 1.772*(cb - 0.5)
    return np.array([r,g,b])



def mse(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat-x_true))
    energy = np.mean(np.square(x_true))    
    return mse/energy

def psnr(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

def num_param(net):
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    return s
    #print('Number of params: %d' % s)

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array([gray])

def savemtx_for_logplot(A,filename = "exp.dat"):
    ind = sorted(list(set([int(i) for i in np.geomspace(1, len(A[0])-1 ,num=700)])))
    A = [ [a[i] for i in ind]  for a in A]
    X = np.array([ind] + A)
    np.savetxt(filename, X.T, delimiter=' ')
