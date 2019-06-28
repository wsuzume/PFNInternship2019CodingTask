import numpy as np

import os
import random

random.seed(0)
np.random.seed(0)

#dataset_path = "../datasets"
#os.listdir("../datasets")

def data_number(s):
    return s.split("_")[0]

def read_graph(path):
    with open(path, 'r') as f:
        lines = list(f.readlines())
        N = int(lines.pop(0).rstrip('\n'))
        G = np.array([list(map(int, line.rstrip('\n').split())) for line in lines])
    return (N, G)

def read_label(path):
    with open(path, 'r') as f:
        line = f.readline()
    return int(line.rstrip('\n'))

def read_dataset(path, train_or_test):
    D = {}
    ds = [p for p in os.listdir(os.path.join(path, train_or_test))]

    if train_or_test == "test":
        xs = sorted([x for x in ds if "graph" in x])
        for x in xs:
            x_num = data_number(x)
            N, G = read_graph(os.path.join(path, train_or_test, x))
            D[int(x_num)] = (N, G)

    elif train_or_test == "train":
        xs = sorted([x for x in ds if "graph" in x])
        ys = sorted([y for y in ds if "label" in y])
        for x, y in zip(xs, ys):
            x_num = data_number(x)
            y_num = data_number(y)
            if x_num != y_num:
                raise ValueError(f"data number not match between '{x}' and '{y}'")
            N, G = read_graph(os.path.join(path, train_or_test, x))
            L = read_label(os.path.join(path, train_or_test, y))
            D[int(x_num)] = (N, G, L)

    else:
        raise ValueError("argument 'train_or_test' must be 'train' or 'test'")

    return D

def dataset_split(dataset):
    vs = list(dataset.values())
    data_list = random.sample(vs, len(vs))

    # use first 400 data for validation
    validate = data_list[:400]
    train = data_list[400:]

    return (train, validate)

def minibatch_split(vs, batch_size):
    data_list = random.sample(vs, len(vs))

    # batch_size must be a divisor of 1600
    N = 1600 // batch_size
    train = []
    for n in range(N):
        train.append(data_list[:batch_size])
        data_list = data_list[batch_size:]

    return train

def measure(conf):
    TP = conf[0, 0]
    FP = conf[0, 1]
    FN = conf[1, 0]
    TN = conf[1, 1]

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (FP + TN)
    return (accuracy, precision, recall, specificity)

def ReLU(x):
    if x < 0:
        return 0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold(x, b=0.5):
    if x > b:
        return 1
    return 0

# this class stores data X by row after the traditional sklearn implemenrations
## 課題 1 ここから =================================
class GNN:
    def __init__(self, W, A, f=ReLU):
        # nonlinear function for combine
        self.f = f

        D = W.shape[0]

        self.n_features = D

        # weight for combine R^{DxD}
        self.W = W

        # weight for sigmoid layer (including bias term) R^{D+1}
        self.A = A

    def aggregate(self, X, G):
        A = np.copy(X)
        X_a = np.copy(X)
        for i, edges in enumerate(G):
            a = np.zeros(A.shape[1])
            for j, e in enumerate(edges):
                if i != j and e == 1:
                    a += A[j]
            X_a[i] = a
        return X_a

    def combine(self, X_a, W=None):
        if W is None:
            W_t = self.W
        else:
            W_t = W

        X_new = X_a.dot(W_t)
        for i in range(X_new.shape[0]):
            for j in range(X_new.shape[1]):
                X_new[i, j] = self.f(X_new[i, j])

        return X_new

    def readout(self, X_new):
        return X_new.sum(axis=0)

    ## 課題 1 ここまで ===============================
    ## 課題 2 ここから ===============================

    def sigmoid_layer(self, readout, A=None):
        if A is None:
            A_t = self.A
        else:
            A_t = A

        hg = np.hstack((readout, [1]))
        s = A_t.dot(hg)
        p = sigmoid(s)
        return (p, s) 

    def threshold(self, p, c=0.5):
        return threshold(p, c)

    # get pair of the predicted probability and value s before sigmoid function
    def predict_prob(self, X, G, W=None, A=None, T=2):
        X_new = X
        for t in range(T):
            X_new = self.combine(self.aggregate(X_new, G), W)

        readout = self.readout(X_new)
        return self.sigmoid_layer(readout, A)

    def predict(self, X, G, W=None, A=None, T=2, c=0.5):
        p, _ = self.predict_prob(X, G, W, A, T)
        return self.threshold(p, c)

    def loss_function(self, y, X, G, W=None, A=None, T=2):
        p, s = self.predict_prob(X, G, W, A, T)

        d = 15
        # this threshold is choosed because e^{15} is big enough
        if s > d:
            right_term = s
        else:
            right_term = np.log(1 + np.exp(s))

        # same reason as above
        if s < -d:
            left_term = -s
        else:
            left_term = np.log(1 + np.exp(-s))

        return y * left_term + (1-y) * right_term

    def batch_loss(self, train_dataset, W=None, A=None, T=2):
        D = self.n_features

        loss = 0
        for N, G, y in train_dataset:
            X = np.zeros((N, D))
            X[:, 0] = np.ones(N)

            loss += self.loss_function(y, X, G, W, A, T)
        return loss

    def calc_gradient(self, train_dataset, eps=1e-3, T=2):
        prev_loss = self.batch_loss(train_dataset, T=T)

        grad_W = np.zeros(self.W.shape)
        W = np.copy(self.W)
        for i in range(grad_W.shape[0]):
            for j in range(grad_W.shape[1]):
                W[i][j] += eps
                loss = self.batch_loss(train_dataset, W=W)
                W[i][j] = self.W[i][j]

                grad_W[i][j] = (loss - prev_loss) / eps

        grad_A = np.zeros(self.A.shape)
        A = np.copy(self.A)
        for i in range(grad_A.shape[0]):
            A[i] += eps
            loss = self.batch_loss(train_dataset, A=A)
            A[i] = self.A[i]

            grad_A[i] = (loss - prev_loss) / eps

        return (grad_W, grad_A, prev_loss)

    def update_parameter(self, W, A):
        self.W = W
        self.A = A

    def gradient_descent(self, train_dataset, alpha=1e-4, eps=1e-3, repeat=100, T=2):
        loss = []

        for t in range(repeat):
            dW, dA, prev_loss = self.calc_gradient(train_dataset, eps, T)
            loss.append(prev_loss)

            W_new = self.W - alpha * dW
            A_new = self.A - alpha * dA
            self.update_parameter(W_new, A_new)

        return loss

    ## 課題 2 ここまで ===============================
    ## 課題 3 ここから ===============================

    def calc_mean_gradient(self, batch, eps=1e-3, T=2):
        batch_size = len(batch)

        dW = np.zeros(self.W.shape)
        dA = np.zeros(self.A.shape)

        for data in batch:
            grad_W, grad_A, loss = self.calc_gradient([data], eps, T)
            dW += grad_W
            dA += grad_A

        dW /= batch_size
        dA /= batch_size

        return dW, dA


    def sgd(self, train, batch_size=10, alpha=1e-4, epoch=1):
        loss = []
        for e in range(epoch):
            # loss on whole training data
            loss.append(self.batch_loss(train))

            minibatch = minibatch_split(train, batch_size)
            for batch in minibatch:
                dW, dA = self.calc_mean_gradient(batch)

                W_new = self.W - alpha * dW
                A_new = self.A - alpha * dA
                self.update_parameter(W_new, A_new)

        return loss

    def momentum_sgd(self, train, batch_size=10, alpha=1e-4, eta=0.9, epoch=1):
        loss = []

        W_moment = np.zeros(self.W.shape)
        A_moment = np.zeros(self.A.shape)

        for e in range(epoch):
            # loss on whole training data
            loss.append(self.batch_loss(train))

            minibatch = minibatch_split(train, batch_size)
            for batch in minibatch:
                dW, dA = self.calc_mean_gradient(batch)

                W_new = self.W - alpha * dW + eta * W_moment
                A_new = self.A - alpha * dA + eta * A_moment

                W_moment = -alpha * dW + eta * W_moment
                A_moment = -alpha * dA + eta * A_moment

                self.update_parameter(W_new, A_new)

        return loss

    ## 課題 3 ここまで ===============================
    ## 課題 4 ここから ===============================

    def adam(self, train_dataset):
        return

    def validate(self, validate_dataset, T=2, c=0.5):
        D = self.W.shape[0]

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for N, G, y in validate_dataset:
            X = np.zeros((N, D))
            X[:, 0] = np.ones(N)

            y_pred = self.predict(X, G, T=T, c=c)

            if y == 1:
                if y == y_pred:
                    TP += 1
                else:
                    FN += 1
            else:
                if y == y_pred:
                    TN += 1
                else:
                    FP += 1

        return np.array([[TP, FP],
                         [FN, TN]])


    def predict_list(self, test_dataset, T=2, c=0.5):
        D = self.W.shape[0]

        ys = []
        for N, G in test_dataset:
            X = np.zeros((N, D))
            X[:, 0] = np.ones(N)

            y_pred = self.predict(X, G, T=T, c=c)
            ys.append(y_pred)

        return ys

    ## 課題 4 ここまで ===============================
