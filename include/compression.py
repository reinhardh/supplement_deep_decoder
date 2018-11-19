from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np

from .helpers import *
from .decoder import *
from .fit import *
from .wavelet import *

def rep_error_deep_decoder(img_np,k=128,convert2ycbcr=False):
    '''
    mse obtained by representing img_np with the deep decoder
    '''
    output_depth = img_np.shape[0]
    if output_depth == 3 and convert2ycbcr:
        img = rgb2ycbcr(img_np)
    else:
        img = img_np
    img_var = np_to_var(img).type(dtype)
    
    num_channels = [k]*5
    net = decodernwv2(output_depth,num_channels_up=num_channels,bn_before_act=True).type(dtype)
    rnd = 500
    numit = 15000
    rn = 0.005
    mse_n, mse_t, ni, net = fit( num_channels=num_channels,
                        reg_noise_std=rn,
                        reg_noise_decayevery = rnd,
                        num_iter=numit,
                        LR=0.004,
                        img_noisy_var=img_var,
                        net=net,
                        img_clean_var=img_var,
                        find_best=True,
                               )
    out_img = net(ni.type(dtype)).data.cpu().numpy()[0]
    if output_depth == 3 and convert2ycbcr:
        out_img = ycbcr2rgb(out_img)
    return psnr(out_img,img_np), out_img, num_param(net)

def rep_error_wavelet(img_np,ncoeff=300):
    '''
    mse obtained by representing img_np with wavelet thresholding
    ncoff coefficients are retained per color channel
    '''
    if img_np.shape[0] == 1:
        img_np = img_np[0,:,:]
        out_img_np = denoise_wavelet(img_np, ncoeff=ncoeff, multichannel=False, convert2ycbcr=True, mode='hard')
    else:
        img_np = np.transpose(img_np)
        out_img_np = denoise_wavelet(img_np, ncoeff=ncoeff, multichannel=True, convert2ycbcr=True, mode='hard')
    # img_np = np.array([img_np[:,:,0],img_np[:,:,1],img_np[:,:,2]])
    return psnr(out_img_np,img_np), out_img_np

def myimgshow(plt,img):
    if(img.shape[0] == 1):
        plt.imshow(np.clip(img[0],0,1),cmap='Greys',interpolation='none')
    else:
        plt.imshow(np.clip(img.transpose(1, 2, 0),0,1),interpolation='none')    
        
    