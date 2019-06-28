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



# weight
W_momentum = np.load("W_momentum.npy")
A_momentum = np.load("A_momentum.npy")

gnet_momentum = gnn.GNN(W_momentum, A_momentum)

test_data_list = []
for idx, data in sorted(test_dataset.items()):
    print(idx)
    test_data_list.append(data)

y_pred = gnet_momentum.predict_list(test_data_list)

with open("prediction.txt", "w") as f:
    for y in y_pred:
        f.write(str(y)+'\n')
