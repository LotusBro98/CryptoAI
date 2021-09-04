from Cryptodome.Cipher import AES
import numpy as np
import matplotlib.pyplot as plt

import distribution
from layer import Layer

N_ALL = 1024 * 4
N_LAYERS = 11
CUT_PROB = 0.001

with open("data/Sample-text-file-1000kb.txt", "rt") as f:
    data = f.read()
data = data[:N_ALL]
data = data.encode()
data0 = data

data = np.frombuffer(data, dtype=np.uint8)

layers = []

for i in range(N_LAYERS):
    layer = Layer()
    layer.fit(data, cut_p=CUT_PROB)
    data = layer.compress(data)
    distribution.test_data(data)
    layers.append(layer)
    print(data.shape)

for i in range(N_LAYERS):
    data = layers[N_LAYERS-1-i].decompress(data)
    print(data.shape)

data = data.tobytes().decode()
print(data)

# key = b'Sixteen byte key'
# cipher = AES.new(key, AES.MODE_EAX)
#
# nonce = cipher.nonce
# ciphertext, tag = cipher.encrypt_and_digest(data0)
# print(ciphertext)
#
# cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
# plaintext = cipher.decrypt(ciphertext)
# print(plaintext)
#
# data = ciphertext
# data = np.frombuffer(data, dtype=np.uint8)
# data = np.unpackbits(data)
# data = np.reshape(data, (-1, 8))
#
# data = data.reshape((-1, 2, data.shape[-1]))
# data = np.split(data, data.shape[-2], axis=-2)
# data = np.concatenate(data, axis=-1)
# print(data.shape)
#
# A = distribution.get_cross_probability_table(data)
# print(A)
# plt.imshow(np.abs(A))
# plt.show()