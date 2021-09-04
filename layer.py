import numpy as np

from distribution import packbits, unpackbits, CHUNK_N

class Layer:
    def fit(self, x, cut_p=0.01):
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
        xb = unpackbits(x)
        px = np.average(xb, axis=0)
        save_bits = np.sum(px > cut_p)
        LUT <<= (m_bits - save_bits)
        LUT >>= (m_bits - save_bits)

        LUT_1 = ind[:2 ** save_bits]

        self.LUT = LUT
        self.LUT_1 = LUT_1
        self.save_bits = save_bits
        self.width = x.shape[-1]

    def compress(self, x):
        y = np.take(self.LUT, x)
        y = unpackbits(y)
        y = y.T[-self.save_bits:].T
        y = y.reshape(y.shape[:-2] + (-1,))
        rest = (y.shape[-1] // CHUNK_N + (y.shape[-1] % CHUNK_N != 0)) * CHUNK_N - y.shape[-1]
        y = np.pad(y, ((0, rest),), constant_values=0)
        y = y.reshape(y.shape[:-1] + (-1, CHUNK_N))
        y = packbits(y)
        return y

    def decompress(self, y):
        y = unpackbits(y)
        y = y.reshape(y.shape[:-2] + (-1,))
        y = np.split(y, np.arange(self.save_bits, y.shape[-1], self.save_bits))
        y = y[:self.width]
        y = np.stack(y, axis=-2)
        y = np.pad(y, ((0,0), (CHUNK_N - self.save_bits, 0)))
        y = packbits(y)
        x = np.take(self.LUT_1, y)
        return x