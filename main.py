from Cryptodome.Cipher import AES
import numpy as np
import matplotlib.pyplot as plt

import distribution
from layer import Layer
from model import Model

N_MAX = 16
MAX_SAMPLES = 10000
N_LAYERS = 10
CUT_PROB = 0.01

with open("data/Sample-text-file-1000kb.txt", "rt") as f:
    data = f.read()
data = data.split(" ")
# data = data[:MAX_SAMPLES]
data = [a + " " + b for a, b in zip(data[0::2], data[1::2])]
data = [(word[:N_MAX] + " " * (N_MAX - len(word))).encode() for word in data]
data0 = data

data = np.frombuffer(b"".join(data), dtype=np.uint8)
data = data.reshape((-1, N_MAX, 1))

layers = []

model = Model()
model.fit(data, CUT_PROB)

latent = model.compress(data)

data = model.decompress(latent)

for i in range(16):
    print("===================")
    print(data0[i].decode())
    print("-------------------")
    print(data[i].tobytes().decode())



# key = b'Sixteen byte key'
#
# data_crypt = []
# for i, word in enumerate(data0):
#     key = word
#     cipher = AES.new(key, AES.MODE_EAX)
#     nonce = cipher.nonce
#     ciphertext, tag = cipher.encrypt_and_digest(data0[0])
#     data_crypt.append(ciphertext)
#     print(f"\r{i} / {len(data0)}", end="")
# data_crypt = np.frombuffer(b"".join(data_crypt), dtype=np.uint8)
# data_crypt = data_crypt.reshape((-1, N_MAX, 1))
# data_crypt = np.unpackbits(data_crypt, axis=-1)
#
# model_crypt = Model()
# model_crypt.fit(data_crypt, cut_p=CUT_PROB)
