import numpy as np

from distribution import CHUNK_N


def packbits(x, n=CHUNK_N, last_only=False):
    rest = (x.shape[-1] // CHUNK_N + (x.shape[-1] % CHUNK_N != 0)) * CHUNK_N - x.shape[-1]
    x = np.pad(x, [(0, 0)] * (len(x.shape) - 1) + [(rest, 0)], constant_values=0)
    if not last_only:
        x = x.reshape(x.shape[:-1] + (-1, CHUNK_N))

    x1, x0 = np.split(x, 2, axis=-1)
    x0 = np.packbits(x0, axis=-1).squeeze(-1)
    x1 = np.packbits(x1, axis=-1).squeeze(-1)
    x = (np.uint16(x1) << 8) + np.uint16(x0)
    return x


def unpackbits(x, n=CHUNK_N, last_only=False):
    x0 = np.uint8(x)
    x1 = np.uint8(x >> 8)

    x0 = np.unpackbits(x0).reshape(x0.shape[:-1] + (-1, 8))
    x1 = np.unpackbits(x1).reshape(x1.shape[:-1] + (-1, 8))
    x = np.concatenate([x1, x0], axis=-1)

    if not last_only:
        x = x.reshape(x.shape[:-2] + (-1,))
    x = x[..., -n:]

    return x


def flatten(x: np.ndarray):
    if len(x.shape) <= 2:
        return x
    return x.reshape((-1, x.shape[-1]))


def convolve1d(x: np.ndarray, ksize=3, stride=2):
    assert((ksize - 1) % 2 == 0)
    x = [x[...,i:x.shape[-2]-(ksize-1-i):stride,:] for i in range(ksize)]
    x = np.concatenate(x, axis=-1)
    return x


def deconvolve1d(x: np.ndarray, ksize=3, stride=2):
    assert ((ksize - 1) % 2 == 0)
    xs = np.split(x, ksize, axis=-1)
    x = np.ones(x.shape[:-2] + (x.shape[-2] * 2 + 1, x.shape[-1] // 3), dtype=np.uint8)
    for i, xi in enumerate(xs):
        x[..., i:x.shape[-2] - (ksize - 1 - i):stride, :] &= xi
    return x
