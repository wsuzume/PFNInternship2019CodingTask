import gnn
import numpy as np
from matplotlib import pyplot as plt

dataset_path = "../datasets"

dataset = gnn.read_dataset(dataset_path, "train")
print("Whole training dataset:", len(dataset))

train_dataset, validate_dataset = gnn.dataset_split(dataset)
print("Splitted train / validate:", len(train_dataset), '/', len(validate_dataset))

test_dataset = gnn.read_dataset(dataset_path, "test")
print("Test dataset:", len(test_dataset))

# dimension of node signal
D = 8

# choose one data you like
N, G, y = dataset[5]

# weight
W = np.random.normal(0, 0.4, (D, D))
A = np.random.normal(0, 0.4, D+1)
A[D] = 0


gnet = gnn.GNN(W, A)
print("start fitting ...")
loss_momentum = gnet.momentum_sgd(train_dataset, batch_size=10, alpha=1e-3, epoch=50)

np.save("W_momentum.npy", gnet.W)
np.save("A_momentum.npy", gnet.A)
np.save("loss_momentum.npy", np.array(loss_momentum))

plt.figure()
plt.plot(loss_momentum)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Fitting with Momentum SGD")
plt.show()
