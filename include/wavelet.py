#import matplotlib.pyplot as plt
import numpy as np
import numbers
import pywt
import scipy
import skimage.color as color
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr
from include import * 

def _wavelet_threshold(image, wavelet, ncoeff = None, threshold=None, mode='soft', wavelet_levels=None):

    wavelet = pywt.Wavelet(wavelet)

    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = [slice(s) for s in image.shape]

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        dlen = wavelet.dec_len
        wavelet_levels = np.min(
            [pywt.dwt_max_level(s, dlen) for s in image.shape])

        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]
    
    a = []
    for level in dcoeffs:
        for key in level:
            a += [np.ndarray.flatten(level[key])]
    a = np.concatenate(a)
    a = np.sort( np.abs(a) )    

    sh = coeffs[0].shape
    basecoeffs = sh[0]*sh[1]
    threshold = a[- (ncoeff - basecoeffs)]
    
    # A single threshold for all coefficient arrays
    denoised_detail = [{key: pywt.threshold(level[key],value=threshold,
                                mode=mode) for key in level} for level in dcoeffs]
   
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]


def denoise_wavelet(image, ncoeff=None, wavelet='db1', mode='hard',
                    wavelet_levels=None, multichannel=False,
                    convert2ycbcr=False):

    image = img_as_float(image)
    
    
    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(image)
            for i in range(3):
                # renormalizing this color channel to live in [0, 1]
                min, max = out[..., i].min(), out[..., i].max()
                channel = out[..., i] - min
                channel /= max - min
                out[..., i] = denoise_wavelet(channel, wavelet=wavelet,ncoeff=ncoeff,
                                              mode=mode,
                                              wavelet_levels=wavelet_levels)

                out[..., i] = out[..., i] * (max - min)
                out[..., i] += min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = _wavelet_threshold(image[..., c],ncoeff=ncoeff,
                                                 wavelet=wavelet, mode=mode,
                                                 wavelet_levels=wavelet_levels)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet, mode=mode,ncoeff=ncoeff,
                                 wavelet_levels=wavelet_levels)

    clip_range = (-1, 1) if image.min() < 0 else (0, 1)
    return np.clip(out, *clip_range)


