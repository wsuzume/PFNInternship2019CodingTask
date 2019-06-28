import unittest
import numpy as np
import gnn

from numpy.testing import assert_array_equal

dataset_path = "../datasets"

class TestGNN(unittest.TestCase):
    def test_read_dataset(self):
        dataset = gnn.read_dataset(dataset_path, "train")
        self.assertEqual(len(dataset), 2000)

        train_dataset, validate_dataset = gnn.dataset_split(dataset)
        self.assertEqual(len(train_dataset), 1600)
        self.assertEqual(len(validate_dataset), 400)

        test_dataset = gnn.read_dataset(dataset_path, "test")
        self.assertEqual(len(test_dataset), 500)

    def test_aggregate(self):
        dataset = gnn.read_dataset(dataset_path, "train")
        train_dataset, validate_dataset = gnn.dataset_split(dataset)
        test_dataset = gnn.read_dataset(dataset_path, "test")

        # adjacency matrix of dataset[0] is
        # array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
        # so when we use R^{11x8} matrix
        # array([[1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 0, 0]])
        # for X, we get
        # array([[ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        # for first aggregation. And we choose
        # array([[ 1.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
        # for W in order to check if 'combine' phase is working well.
        # When we use this W, we get
        # array([[ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.],
        #        [ 4.,  4.,  0.,  0.,  0.,  0.,  0.,  0.]])
        # for updated X because X[:, 1] is copied from X[:, 0], and
        # X[:, 2] is copied from X[:, 0] too but the sign be inverted
        # so X[:, 2] must be vanished because of ReLU.
        # Then we get
        # array([ 18., 18.,  0.,  0.,  0.,  0.,  0.,  0.])
        # for READOUT.

        # dimension of node signal
        D = 8

        # number of nodes
        N = dataset[0][0]

        # adjacency matrix
        G = dataset[0][1]

        # weight
        W = np.eye(D)
        W[0, 1] = 1
        W[0, 2] = -1

        # we don't use this parameter in this test
        A = np.ones(D+1)
        A[D] = 0

        # signal on nodes
        X = np.zeros((N, D))
        X[:, 0] = np.ones(N)

        gnet = gnn.GNN(W, A)

        assert_array_equal(gnet.aggregate(X, G),
            np.array([[ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]))
        assert_array_equal(gnet.combine(gnet.aggregate(X, G), W),
            np.array([[ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [ 4.,  4.,  0.,  0.,  0.,  0.,  0.,  0.]]))
        assert_array_equal(gnet.readout(gnet.combine(gnet.aggregate(X, G), W)),
            np.array([ 18., 18.,  0.,  0.,  0.,  0.,  0.,  0.]))
