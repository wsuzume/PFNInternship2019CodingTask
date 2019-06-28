import gnn
import numpy as np

#np.random.seed(0)

dataset_path = "../datasets"

dataset = gnn.read_dataset(dataset_path, "train")
print("Whole training dataset:", len(dataset))

train_dataset, validate_dataset = gnn.dataset_split(dataset)
print("Splitted train / validate:", len(train_dataset), '/', len(validate_dataset))

test_dataset = gnn.read_dataset(dataset_path, "test")
print("Test dataset:", len(test_dataset))

# dimension of node signal
D = 8

# number of nodes
N = dataset[0][0]

# adjacency matrix
G = dataset[0][1]

# weight
#W = np.eye(D)
W = np.random.normal(0, 0.4, (D, D))
#A = np.ones(D+1)
A = np.random.normal(0, 0.4, D+1)
A[D] = 0

# label
y = dataset[0][2]

# signal on nodes
X = np.zeros((N, D))
X[:, 0] = np.ones(N)
#X = np.ones((N, D))
#for i in range(X.shape[0]):
#    X[i] *= 2 ** i
#X = np.random.normal(10, 5, (N, D))

gnet = gnn.GNN(W, A)
print(gnet.aggregate(X, G))
print(gnet.combine(gnet.aggregate(X, G), W))
print(gnet.readout(gnet.combine(gnet.aggregate(X, G), W)))
