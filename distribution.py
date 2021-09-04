import numpy as np
import matplotlib.pyplot as plt

CHUNK_N = 16

def flatten(x: np.ndarray):
    if len(x.shape) <= 2:
        return x
    return x.reshape((-1, x.shape[-1]))

def unpackbits(x):
    x0 = np.uint8(x)
    x1 = np.uint8(x >> 8)

    x0 = np.unpackbits(x0).reshape(x0.shape[:-1] + (-1, 8))
    x1 = np.unpackbits(x1).reshape(x1.shape[:-1] + (-1, 8))
    x = np.concatenate([x1, x0], axis=-1)
    return x

def packbits(x):
    x1, x0 = np.split(x, 2, axis=-1)
    x0 = np.packbits(x0)
    x1 = np.packbits(x1)
    x = (np.uint16(x1) << 8) + np.uint16(x0)
    return x

def get_cross_probability_table(x):
    x = unpackbits(x)
    x = np.reshape(x, (-1, CHUNK_N))[:,::-1]
    px = np.average(x, axis=0)
    xx = x[:,np.newaxis,:] * x[:,:,np.newaxis]
    pxx = np.average(xx, axis=0)
    a = pxx / (px[np.newaxis,:] * px[:,np.newaxis])
    a = np.log(a)
    a[np.isnan(a) | np.isinf(a)] = 0
    a[np.eye(a.shape[-1], dtype=np.bool)] = 0

    print(px)
    return a

def test_data(x):
    A = get_cross_probability_table(x)
    plt.imshow(np.abs(A))
    plt.show()
    print(np.linalg.norm(A))