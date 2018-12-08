from TreeModel.Tree import *
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
from TestRunnables.test_run_methods import test_run_tree

DEPTH = 3

iris = load_breast_cancer()
X = iris.data
y = iris.target

t = Tree(max_depth=DEPTH)

(y_pred, time_train, time_test) = test_run_tree(t,X,y)

print(accuracy_score(y, y_pred))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=DEPTH)

(y_pred2, time_train2, time_test2) = test_run_tree(clf,X,y)

print(accuracy_score(y, y_pred2))

print("Custom alg. time to train:", time_train)
print("Custom alg. time to test:", time_test)
print("SKLearn alg. time to train:", time_train2)
print("SKLearn alg. time to test:", time_test2)

dot_data = export_graphviz(clf, out_file=None,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("treeGraph")
