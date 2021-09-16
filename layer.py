import numpy as np
import matplotlib.pyplot as plt

import distribution
from distribution import CHUNK_N
from operations import packbits, unpackbits, convolve1d, deconvolve1d


class Layer:
    def __init__(self, convolve=False):
        self.convolve = convolve

    def fit(self, x, cut_p=0.01):
        self.width, self.depth = x.shape[-2:]
        if self.width == 1:
            self.convolve = False
        x = packbits(x)

        self.LUT = []
        self.LUT_1 = []
        self.save_bits = []
        for i in range(x.shape[-1]):
            xi = x[..., i]
            LUT, LUT_1, save_bits = distribution.get_optimal_LUT(xi, cut_p)
            self.LUT.append(LUT)
            self.LUT_1.append(LUT_1)
            self.save_bits.append(save_bits)
        self.save_bits = np.stack(self.save_bits, axis=0)

    def compress(self, x):
        x = packbits(x)
        y = np.concatenate([
            unpackbits(np.take(self.LUT[i], x[..., i]), n=self.save_bits[i], last_only=True)
            for i in range(x.shape[-1])
        ], axis=-1)

        if (self.convolve):
            y = convolve1d(y)
        return y

    def decompress(self, y):
        if (self.convolve):
            y = deconvolve1d(y)

        y = np.split(y, np.cumsum(self.save_bits), axis=-1)[:-1]
        x = [np.take(self.LUT_1[i], packbits(yi, last_only=True)) for i, yi in enumerate(y) if yi.shape[-1] > 0]
        x = np.stack(x, axis=-1)
        x = unpackbits(x, n=self.depth)

        return x
