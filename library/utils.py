import os
from glob import glob
import numpy as np
from PIL import Image 

def normalize(data, norm_type="min_max"):
    # noramlize the data as zero to one
    if norm_type == "min_max":
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = np.clip(data, 0, 1)

    elif norm_type == "002_098":
        data = (data - np.quantile(data, 0.02)) / (np.quantile(data, 0.98) - np.quantile(data, 0.02))
        data = np.clip(data, 0, 1)

    elif norm_type == "002_095":
        data = (data - np.quantile(data, 0.02)) / (np.quantile(data, 0.95) - np.quantile(data, 0.02))
        data = np.clip(data, 0, 1)

    else:
        print("error")
        exit()

    return data

def to_gif_and_save(frames_np, save_name):

    frames_np = np.array(frames_np * 255.0, dtype=np.uint8)

    frames = [Image.fromarray(frame, mode="L") for frame in frames_np]

    frames[0].save(save_name, save_all=True, append_images=frames[1:], duration=400, loop=0)

def temporal_lowpass_iir(frames_np: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply 1st-order IIR low-pass filter (EMA) along the temporal (frame) axis.
    
    Args:
        frames_np (np.ndarray): shape (T, H, W), where T = frame count.
        alpha (float): smoothing factor (0 < alpha < 1).
                       Lower means smoother (more filtering), higher means more responsive.

    Returns:
        np.ndarray: filtered array of shape (T, H, W)
    """
    assert frames_np.ndim == 3, "Input must be [T, H, W]"
    T, H, W = frames_np.shape
    filtered = np.zeros_like(frames_np)
    
    # 초기화: 첫 프레임은 그대로
    filtered[0] = frames_np[0]

    # IIR filtering along time axis
    for t in range(1, T):
        filtered[t] = alpha * frames_np[t] + (1 - alpha) * filtered[t - 1]

    return filtered

import numpy as np

def fft_temporal_bandpass(frames_np: np.ndarray, low_cutoff: float = 0.05, high_cutoff: float = 0.4) -> np.ndarray:
    """
    Apply FFT-based band-pass filter along the temporal axis of a [T, H, W] tensor.

    Args:
        frames_np (np.ndarray): Input array of shape (T, H, W)
        low_cutoff (float): Lower bound of frequency band (e.g., 0.05)
        high_cutoff (float): Upper bound of frequency band (e.g., 0.4)

    Returns:
        np.ndarray: Filtered array of same shape
    """
    assert frames_np.ndim == 3, "Input must be [T, H, W]"
    assert 0 <= low_cutoff < high_cutoff <= 0.5, "Cutoffs must be in (0, 0.5] and low < high"

    T, H, W = frames_np.shape
    fft = np.fft.fft(frames_np, axis=0)  # [T, H, W]

    freqs = np.fft.fftfreq(T)  # [T]
    # Band-pass mask: keep only frequencies within [low_cutoff, high_cutoff]
    mask = (np.abs(freqs) >= low_cutoff) & (np.abs(freqs) <= high_cutoff)
    mask = mask[:, None, None]  # reshape to [T, 1, 1] for broadcasting

    fft_filtered = fft * mask
    filtered = np.fft.ifft(fft_filtered, axis=0).real  # remove imag part

    return filtered
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:19:12 2015

@author: jkooij
"""


class Pyramid2arr:
    '''Class for converting a pyramid to/from a 1d array'''
    
    def __init__(self, steer, coeff=None):
        """
        Initialize class with sizes from pyramid coeff
        """
        self.levels = range(1, steer.height-1)
        self.bands = range(steer.nbands)
        
        self._indices = None
        if coeff is not None:
            self.init_coeff(coeff)

    def init_coeff(self, coeff):       
        shapes = [coeff[0].shape]        
        for lvl in self.levels:
            for b in self.bands:
                shapes.append( coeff[lvl][b].shape )             
        shapes.append(coeff[-1].shape)

        # compute the total sizes        
        sizes = [np.prod(shape) for shape in shapes]
        
        # precompute indices of each band
        offsets = np.cumsum([0] + sizes)
        self._indices = list(zip(offsets[:-1], offsets[1:], shapes))

    def p2a(self, coeff):
        """
        Convert pyramid as a 1d Array
        """
        
        if self._indices is None:
            self.init_coeff(coeff)
        
        bandArray = np.hstack([ np.ravel( coeff[lvl][b] ) for lvl in self.levels for b in self.bands ])
        bandArray = np.hstack((np.ravel(coeff[0]), bandArray, np.ravel(coeff[-1])))

        return bandArray        
        
       
    def a2p(self, bandArray):
        """
        Convert 1d array back to Pyramid
        """
        
        assert self._indices is not None, 'Initialize Pyramid2arr first with init_coeff() or p2a()'

        # create iterator that convert array to images
        it = (np.reshape(bandArray[istart:iend], size) for (istart,iend,size) in self._indices)
        
        coeffs = [next(it)]
        for lvl in self.levels:
            coeffs.append([next(it) for band in self.bands])
        coeffs.append(next(it))

        return coeffs

def find_token_indices(tokens, keywords):
    """토크나이저 출력 토큰 리스트에서 키워드(부분 일치) 인덱스 모음."""
    kws = [k.lower() for k in keywords]
    idxs = []
    for i, t in enumerate(tokens):
        tn = t.lower().replace("▁", "").replace("</w>", "")
        if any(k in tn for k in kws):
            idxs.append(i)
    return sorted(set(idxs))