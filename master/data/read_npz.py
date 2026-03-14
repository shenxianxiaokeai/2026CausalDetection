import numpy as np

theta = np.load("output/main/GN-ipw/theta_GN_time4_ipw_K4.npz")

print(theta.files)

Q = theta["Q"]
B = theta["B"]
Delta = theta["Delta"]
A_shared = theta["shared_adj"]

print("Q shape:", Q.shape)
print("B shape:", B.shape)
print("Delta shape:", Delta.shape)
print("A_shared shape:", A_shared.shape)

print("Q row sum min/max:", Q.sum(axis=1).min(), Q.sum(axis=1).max())
print("Community sizes:", np.bincount(Q.argmax(axis=1)))

print("B:")
print(B)

print("Delta abs mean per layer:", np.mean(np.abs(Delta), axis=(1,2)))
print("A_shared min/max/mean:", A_shared.min(), A_shared.max(), A_shared.mean())