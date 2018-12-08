from sklearn.tree import DecisionTreeClassifier
import numpy as np
from random import randint, seed

class Node:
    def __init__(self, feature_index = None, IG_score = None, current_depth = 0,
                 threshold = None, random_feat = False, tree = None, parent = None):
        self.feature_index = feature_index
        self.IG_score = IG_score
        self.threshold = threshold
        self.depth = current_depth
        self.random = random_feat
        self._tree = tree
        self.child_left = None
        self.child_right = None
        self.parent = parent
        self.class_value = None
        self.posProb = None
        seed()

    def setLeftChild(self, node):
        self.child_left = node

    def setRightChild(self, node):
        self.child_right = node

    def isLeaf(self):
        return self.child_right is None and self.child_left is None

    def setClass(self, class_value):
        self.class_value = class_value

    def _getMajorityClass(self, y_data):
        (vals, counts) = np.unique(y_data, return_counts=True)
        return vals[np.argmax(counts)]

    def _setPosProb(self, y_data):
        if self._tree.isBinaryClassifier():
            nr_pos = 0
            nr_neg = 0
            for i in y_data:
                if i > 0:
                    nr_pos += 1
                else:
                    nr_neg += 1

            self.posProb = nr_pos/(nr_neg+nr_pos)

    def getIG(self, X_data, y_data):
        self.selectFeatureByScore(X_data, y_data)
        if self.feature_index is None:
            return 0
        return self.IG_score

    def train(self, X_data, y_data, max_depth):
        if self.depth == max_depth:
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
        elif all(y_data[number] == y_data[0] for number in range(1, len(y_data) - 1)):
            self.setClass(y_data[0][0])
            self._setPosProb(y_data)
        else:
            # FEATURE AND THRESHOLD SELECTION
            self.selectFeature(X_data, y_data)

            if self.feature_index is None:
                self.setClass(self._getMajorityClass(y_data))
                self._setPosProb(y_data)
                return
            # FEATURE AND THRESHOLD SELECTION END

            # split data and train children
            split_data = FilterData(X_data, y_data, self.threshold,
                                    self.feature_index)

            self.child_left = Node(current_depth=self.depth + 1, random_feat=self.random,
                                   tree=self._tree, parent=self)
            self.child_right = Node(current_depth=self.depth + 1, random_feat=self.random,
                                    tree=self._tree, parent=self)
            self.child_left.train(split_data['leftExamples'], split_data['leftLabels'],
                                  max_depth=max_depth)
            self.child_right.train(split_data['rightExamples'], split_data['rightLabels'],
                                   max_depth=max_depth)

    def selectFeature(self, X_data, y_data):
        if self.random:
            return self.selectFeatureByRandom(X_data, y_data)
        else:
            return self.selectFeatureByScore(X_data, y_data)

    def selectThreshold(self, X_data, y_data, feature_index):
        # if all X data for this feature are equal, then we can't split
        if all(X_data[:, self.feature_index] == X_data[0, self.feature_index]):
            self.setClass(self._getMajorityClass(y_data))
            self._setPosProb(y_data)
            return

        # if we are here then a proper split can be made
        self.threshold = getNodeSplitThreshold(X_data[:, self.feature_index], y_data)

    def selectFeatureByScore(self, X_data, y_data, criterion="entropy"):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=1)
        clf.fit(X_data, y_data)
        if clf.tree_.impurity[0] == 0:
            self.IG_score = None
            self.feature_index = None
            self.threshold = None

        self.feature_index = clf.tree_.feature[0]
        self.threshold = clf.tree_.threshold[0]
        N_left = len(y_data[X_data[:, self.feature_index] <= self.threshold])
        N_right = len(y_data[X_data[:, self.feature_index] > self.threshold])
        N_t = X_data.shape[0]

        self.IG_score = clf.tree_.impurity[0] -(N_left /N_t * clf.tree_.impurity[1]
                                              + N_right/N_t * clf.tree_.impurity[2])

    def selectFeatureByRandom(self, X_data, y_data):
        self.IG_score = -1 # -1 for random criteria selection
        self.feature_index = randint(0, X_data.shape()[1]-1)
        self.threshold = getNodeSplitThreshold(X_data[:,self.feature_index], y_data)

    def predictData(self, data):
        if self.isLeaf():
            if self.class_value is None:
                raise("A class value has not been set to leaf node")
            return self.class_value

        if data[self.feature_index] <= self.threshold:
            return self.child_left.predictData(data)
        else:
            return self.child_right.predictData(data)

    def getPositiveProb(self, data):
        if self.isLeaf():
            return self.posProb

        if data[self.feature_index] <= self.threshold:
            return self.child_left.getPositiveProb(data)
        else:
            return self.child_right.getPositiveProb(data)

    def printNode(self):
        if not self.isLeaf():
            print("feature:", self.feature_index, "threshold:", self.threshold, "IG:", self.IG_score)
        else:
            print("leaf w class:", self.class_value)
            return

        self.child_left.printNode()
        self.child_right.printNode()

    def getClassProb(self, data):
        if self.isLeaf():
            return

# Process each feature using the decision stump model
# Input: feature data, labels
# Return: Predicted value by the model
def getNodeSplitThreshold(samples, labels):
    clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    samples = samples.reshape(-1, 1)
    clf.fit(samples, labels)
    return clf.tree_.threshold[0]

# Split X data by threshold and feature
# Input: predicted labels
# Return: Dictionary with the left and right sub-sets (split the master data)
def FilterData(x_data, y_data, threshold, feature_index):
    x_res_left = np.vstack(x_data[x_data[:,feature_index] <= threshold])
    y_res_left = np.vstack(y_data[x_data[:,feature_index] <= threshold])
    x_res_right = np.vstack(x_data[x_data[:,feature_index] > threshold])
    y_res_right = np.vstack(y_data[x_data[:,feature_index] > threshold])

    return {"leftExamples":x_res_left, "leftLabels":y_res_left,
            "rightExamples":x_res_right, "rightLabels":y_res_right}

