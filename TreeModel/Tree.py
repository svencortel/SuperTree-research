from TreeModel.Node import Node

import numpy as np

class Tree:
    """The structure of the tree"""
    def __init__(self, criterion = "entropy", max_depth = None, lookahead = False,
                 random_feat=False):
        self.criterion = criterion
        self.max_depth = max_depth
        self.lookahead = lookahead
        self.X_data = None
        self.y_data = None
        self.root_node = None
        self.random = random_feat
        self.nr_features = 0
        self._nr_examples = 0

    def load(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.nr_features = X_data.shape[1]
        self._nr_examples = X_data.shape[0]

    def fit(self, X_data, y_data):
        self.train(X_data, y_data)

    def train(self, X_data = None, y_data = None):
        if X_data is None and y_data is None:
            if self.X_data is None:
                raise("No data loaded")
        else:
            self.load(X_data, y_data)

        # init root node and start training
        #print("Training Tree Model. Please Wait...")
        self.root_node = Node(random_feat = self.random, tree=self)

        self.root_node.train(self.X_data, self.y_data, self.max_depth)

    def predict(self, X_data):
        result = []
        # make this more efficient
        for i in X_data:
            result.append(self.root_node.predictData(i))
        return np.array(result)

    def isBinaryClassifier(self):
        return len(np.unique(self.y_data))==2

    def getClassProb(self, X_data):
        if not self.isBinaryClassifier():
            raise Exception("classification must be binary for getClassProb")
        result = []
        for i in X_data:
            result.append(self.root_node.getPositiveProb(i))
        return result

    def printTree(self):
        self.root_node.printNode()
