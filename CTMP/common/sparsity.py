import utilities
import numpy as np

theta = np.load("./theta.npy")

def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    for d in range(batch_size):
        sparsity[d] = len(np.where(doc_tp[d] > 1e-20)[0])
    sparsity /= num_topics
    return np.mean(sparsity)

sparsity = compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')

print(sparsity)