import numpy as np
import matplotlib.pyplot as plt

CHUNK_N = 16

from operations import unpackbits


def get_cross_probability_table(x):
    x = np.reshape(x, (-1, x.shape[-1]))[..., ::-1]
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


def get_optimal_LUT(x, cut_p=0):
    x = x.reshape((-1,))
    m_bits = CHUNK_N
    n_bins = 2 ** m_bits
    px = np.bincount(x, minlength=n_bins)
    px = np.float32(px) / x.shape[0]
    ind = np.uint16(range(n_bins))

    for i in range(m_bits):
        ind_sort = np.argsort(px, axis=-1)
        ind = np.take_along_axis(ind, ind_sort, axis=-1)
        px = np.take_along_axis(px, ind_sort, axis=-1)

        if i == m_bits - 1:
            break

        ind = np.stack(np.split(ind, 2, -1)[::-1], -2)
        px = np.stack(np.split(px, 2, -1)[::-1], -2)

    ind = np.reshape(ind, (-1,))
    LUT = np.zeros((n_bins,), dtype=np.uint16)
    LUT[ind] = np.uint16(range(n_bins))

    x = np.take(LUT, x)
    xb = unpackbits(x, last_only=True)
    px = np.average(xb, axis=0)
    save_bits = np.sum(px > cut_p)
    LUT <<= (m_bits - save_bits)
    LUT >>= (m_bits - save_bits)

    LUT_1 = ind[:2 ** save_bits]

    return LUT, LUT_1, save_bits
