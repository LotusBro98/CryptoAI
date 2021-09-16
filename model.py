from layer import Layer
import numpy as np

class Model:
    def __init__(self):
        self.layers = []

    def fit(self, x, cut_p):
        x = np.unpackbits(x, axis=-1)

        while x.shape[-2] > 1:
            depth_prev = 0
            while depth_prev != x.shape[-1]:
                depth_prev = x.shape[-1]

                layer = Layer(convolve=False)
                layer.fit(x, cut_p)
                self.layers.append(layer)
                x = layer.compress(x)
                print(x.shape)

            layer = Layer(convolve=True)
            layer.fit(x, cut_p)
            self.layers.append(layer)
            x = layer.compress(x)
            print(x.shape)

        depth_prev = 0
        while depth_prev != x.shape[-1]:
            depth_prev = x.shape[-1]

            layer = Layer(convolve=False)
            layer.fit(x, cut_p)
            self.layers.append(layer)
            x = layer.compress(x)
            print(x.shape)

        x = x.squeeze(-2)
        return x

    def compress(self, x):
        x = np.unpackbits(x, axis=-1)

        for layer in self.layers:
            x = layer.compress(x)
        x = x.squeeze(-2)
        return x

    def decompress(self, x):
        x = np.expand_dims(x, axis=-2)
        for layer in self.layers[::-1]:
            x = layer.decompress(x)

        x = np.packbits(x, axis=-1)
        return x
