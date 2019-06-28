import gnn
import numpy as np
from matplotlib import pyplot as plt


def average_accuracy(conf):
    TP = conf[0, 0]
    FP = conf[0, 1]
    FN = conf[1, 0]
    TN = conf[1, 1]

    return (TP / (TP + FN) + TN / (TN + FP)) / 2

dataset_path = "../datasets"

dataset = gnn.read_dataset(dataset_path, "train")
print("Whole training dataset:", len(dataset))

train_dataset, validate_dataset = gnn.dataset_split(dataset)
print("Splitted train / validate:", len(train_dataset), '/', len(validate_dataset))

test_dataset = gnn.read_dataset(dataset_path, "test")
print("Test dataset:", len(test_dataset))



# weight
W_sgd = np.load("W_sgd.npy")
A_sgd = np.load("A_sgd.npy")

W_momentum = np.load("W_momentum.npy")
A_momentum = np.load("A_momentum.npy")

gnet_sgd = gnn.GNN(W_sgd, A_sgd)
gnet_momentum = gnn.GNN(W_momentum, A_momentum)

# mean loss
sgd_loss_t = gnet_sgd.batch_loss(train_dataset) / len(train_dataset)
sgd_loss = gnet_sgd.batch_loss(validate_dataset) / len(validate_dataset)
momentum_loss_t = gnet_momentum.batch_loss(train_dataset) / len(train_dataset)
momentum_loss = gnet_momentum.batch_loss(validate_dataset) / len(validate_dataset)

# confusion matrix
conf_sgd_t = gnet_sgd.validate(train_dataset)
conf_momentum_t = gnet_momentum.validate(train_dataset)

conf_sgd = gnet_sgd.validate(validate_dataset)
conf_momentum = gnet_momentum.validate(validate_dataset)

# measurement
m_sgd_t = gnn.measure(conf_sgd_t)
m_momentum_t = gnn.measure(conf_momentum_t)

m_sgd = gnn.measure(conf_sgd)
m_momentum = gnn.measure(conf_momentum)

print("-------------------------------------")
print("SGD measurement (train)")
print("Average Loss:", sgd_loss_t)
print("Average Accuracy:", average_accuracy(conf_sgd_t))
print("Confusion Matrix:")
print(conf_sgd_t)
print("     Accuracy:", m_sgd_t[0])
print("    Precision:", m_sgd_t[1])
print("       Recall:", m_sgd_t[2])
print("  Specificity:", m_sgd_t[3])
print("-------------------------------------")
print("SGD measurement (validate)")
print("Average Loss:", sgd_loss)
print("Average Accuracy:", average_accuracy(conf_sgd))
print("Confusion Matrix:")
print(conf_sgd)
print("     Accuracy:", m_sgd[0])
print("    Precision:", m_sgd[1])
print("       Recall:", m_sgd[2])
print("  Specificity:", m_sgd[3])
print("-------------------------------------")
print("Momentum measurement (train)")
print("Average Loss:", momentum_loss_t)
print("Average Accuracy:", average_accuracy(conf_momentum_t))
print("Confusion Matrix:")
print(conf_momentum_t)
print("     Accuracy:", m_momentum_t[0])
print("    Precision:", m_momentum_t[1])
print("       Recall:", m_momentum_t[2])
print("  Specificity:", m_momentum_t[3])
print("-------------------------------------")
print("Momentum measurement (validate)")
print("Average Loss:", momentum_loss)
print("Average Accuracy:", average_accuracy(conf_momentum))
print("Confusion Matrix:")
print(conf_momentum)
print("     Accuracy:", m_momentum[0])
print("    Precision:", m_momentum[1])
print("       Recall:", m_momentum[2])
print("  Specificity:", m_momentum[3])
print("-------------------------------------")

loss_sgd = np.load("loss_sgd.npy") / 1600
loss_momentum = np.load("loss_momentum.npy") / 1600

plt.figure(figsize=(14,4))
plt.subplot(1,2,1)
plt.plot(loss_sgd)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Fitting with SGD")
plt.subplot(1,2,2)
plt.plot(loss_momentum)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Fitting with Momentum SGD")
plt.show()
