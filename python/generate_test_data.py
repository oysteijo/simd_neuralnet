import numpy as np
np.random.seed(42)

def make_initial_weights(sizes):
    weights = [[np.random.randn( inp, out ).astype(np.float32) , np.zeros(out, dtype=np.float32)] for inp,out in zip(sizes[:-1], sizes[1:])]
    return tuple([item for sublist in weights for item in sublist])

sizes = [32,16,8,4]
initial_weights = make_initial_weights(sizes)
np.savez("initial_weights.npz", *initial_weights )

inp = np.random.rand(1,sizes[0]).astype(np.float32)
np.save("test_sample.npy", inp)

target = np.array([1.0] + [0.0]*(sizes[-1]-1)).astype(np.float32) # First element 1.0 and the rest 0.0
np.save("test_target.npy", target)

# Use this with MAPE - MAPE with targets containing 0.0 will divide by zero
# and things will be numerically unstable
target_mape = np.array([0.5] *(sizes[-1])).astype(np.float32) # All element 0.5
np.save("test_target_mape.npy", target_mape)

