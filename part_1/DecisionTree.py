"""Decision Tree Classifier."""
import os
from typing import List, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


def entropy(y: np.ndarray) -> float:
    """
    Calculate the entropy of a label array.

    Parameters
    ----------
    y : np.ndarray
        The label array, shape = [n_samples, ]

    Returns
    -------
    float
        The entropy of the label array. The value is >= 0, the smaller the purer.
    """
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))


class DecisionTreeClassifier:
    """
    Implement a simple decision tree classifier based on ID3.
    """

    def __init__(self) -> None:
        """
        Initialize the decision tree classifier.

        The decision "tree" is stored as a dict like {feature_index: {feature_value: sub_tree}}.
        """
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            The training input samples, shape = [n_samples_train, n_features]
        y : np.ndarray
            The target values (class labels), shape = [n_samples_train, ]
        """
        self.tree = self.TreeGenerate(X, y, list(range(X.shape[1])))

    def TreeGenerate(self, X: np.ndarray, y: np.ndarray, attr_list: List[int]) -> Union[int, dict]:
        """
        Generate the decision tree.Refer to the pseudo code in the textbook 《Machine Learning》 by Zhou Zhihua.

        Parameters
        ----------
        X : np.ndarray
            The training input samples, shape = [n_samples_train, n_features]
        y : np.ndarray
            The target values (class labels), shape = [n_samples_train, ]
        attr_list : list
            The list of attributes, subset of [n_features,]

        Returns
        -------
        tree
            The generated decision tree, if the node is a leaf node, return the class label.
        """
        # if all samples are in the same class C, label the node with C
        if len(np.unique(y)) == 1:
            return y[0]
        # if the attribute list is empty OR all samples have the same attribute value, label the node with the most common class in y
        if attr_list is None or len(np.unique(X, axis=0)) == 1:
            return np.argmax(np.bincount(y))
        # select the best attribute to split, default ID3 method
        best_attr = self.SelectBestAttribute(X, y, attr_list)
        tree = {best_attr: {}}
        # split the dataset according to the best attribute
        for value in np.unique(X[:, best_attr]):
            sub_X = X[X[:, best_attr] == value]
            sub_y = y[X[:, best_attr] == value]
            sub_attr_list = attr_list.copy()
            sub_attr_list.remove(best_attr)
            tree[best_attr][value] = self.TreeGenerate(
                sub_X, sub_y, sub_attr_list)
        return tree

    def SelectBestAttribute(self, X: np.ndarray, y: np.ndarray, attr_list: List[int]) -> int:
        """
        Select the best attribute to split the dataset based on information gain.(ID3 method)

        Parameters
        ----------
        X : np.ndarray
            The training input samples, shape = [n_samples_train, n_features]
        y : np.ndarray
            The target values (class labels), shape = [n_samples_train, ]
        attr_list : list
            The list index of attributes, subset of [n_features,]

        Returns
        -------
        int
            The index of the best attribute.
        """
        if len(attr_list) == 0:
            raise ValueError("The attribute list is empty.")
        best_attr = -1
        max_info_gain = 0
        base_entropy = entropy(y)
        for attr in attr_list:
            attr_entropy = 0
            feature_values = np.unique(X[:, attr])
            for value in feature_values:
                sub_y = y[X[:, attr] == value]
                attr_entropy += len(sub_y) / len(y) * entropy(sub_y)
            info_gain = base_entropy - attr_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = attr
        return best_attr

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : np.ndarray
            The input samples, shape = [n_samples_test, n_features]

        Returns
        -------
        y : np.ndarray
            The predicted classes, shape = [n_samples_test, ]
        """
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = self.PredictOneSample(X[i], self.tree)
        return y

    def PredictOneSample(self, sample: np.ndarray, tree: Union[dict, int]) -> int:
        """
        Predict the class label for one sample.

        Parameters
        ----------
        sample : np.ndarray
            The input sample, shape = [n_features, ]
        tree : dict
            The decision tree.

        Returns
        -------
        int
            The predicted class label.
        """
        if not isinstance(tree, dict):
            return tree
        root = list(tree.keys())[0]  # the root node
        root_value = sample[root]
        if root_value not in tree[root]:
            # if no such value in the tree, return the most common class in the subtree
            candidates = []
            for key in tree[root].keys():
                if isinstance(tree[root][key], dict):
                    candidates.append(self.PredictOneSample(
                        sample, tree[root][key]))
                else:
                    candidates.append(tree[root][key])
            return max(candidates, key=candidates.count)
        return self.PredictOneSample(sample, tree[root][root_value])

    def __repr__(self) -> str:
        return "DecisionTreeClassifier"

    def print_tree(self, tree, depth=0):
        if not isinstance(tree, dict):
            print(f"{tree}")
            return
        for key in tree.keys():
            print(f"{'| ' * depth}{key}")
            for value in tree[key].keys():
                print(f"{'| ' * (depth+1)}{value}")
                self.print_tree(tree[key][value], depth+2)


def load_data(datapath: str = './data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"File not found: {datapath}")
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight',
                         'NCP', 'CH2O', 'FAF', 'FCVC', 'TUE']
    discrete_features = ['Gender', 'CALC', 'FAVC',  'SCC', 'SMOKE',
                         'family_history_with_overweight', 'CAEC', 'MTRANS']

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)
    # binarize continue features
    for col in continue_features:
        mid = (X[col].max() + X[col].min())/2
        X[col] = X[col].apply(lambda x: 1 if x > mid else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # My IDE Workd Directory
    data_path = os.path.join(
        os.getcwd(), './data/ObesityDataSet_raw_and_data_sinthetic.csv')
    X_train, X_test, y_train, y_test = load_data(
        datapath=data_path)
    X_train, X_test = X_train.values, X_test.values
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # clf.print_tree(clf.tree)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"accuracy: {acc:.4f}")
