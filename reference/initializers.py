import numpy as np

def xavier(m,h):
    return (np.random.rand(m,h).astype(np.float32) * 2.0 - 1.0)  * np.sqrt(6.0/(m+h)).astype(np.float32)

def kaiming(m,h):
    return np.random.randn(m,h).astype(np.float32) * np.sqrt(2.0/m).astype(np.float32)

def make_initial_weights(sizes, initialisers):
    weights = [[init( inp, out ), np.zeros(out, dtype=np.float32)] for inp,out,init in zip(sizes[:-1], sizes[1:], initialisers)]
    return tuple([item for sublist in weights for item in sublist])
