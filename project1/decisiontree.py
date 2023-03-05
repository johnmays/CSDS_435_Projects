from classifier import Classifier

import math
import numpy as np
from scipy import stats
from tqdm import tqdm
from util import entropy

class Node:
    def __init__(self, branch_factor=4):
        self.branch_factor = branch_factor
        self.children = []
        self.label = 0

        self.attribute = 0

    def route(self, testing_sample: np.ndarray) -> int:
        if len(self.children) == 0:
            return self.label
        child_index = int(np.floor(testing_sample[self.attribute] / (256/self.branch_factor)))
        return self.children[child_index].route(testing_sample)


class DecisionTree(Classifier):
    def __init__(self, branch_factor=2, max_depth=8):
        self.root = None

        self.branch_factor = branch_factor
        self.max_depth = max_depth

    @property
    def name(self):
        return "Decision Tree"

    def populate_node(self, node: Node, depth: int, X: np.ndarray, y: np.ndarray) -> None:
        """
        Given the set of training examples and labels that would flow to a node,
        find the attribute that maximizes information gain on the remaining examples,
        set the node to partition on that attribute,
        and call this function recursively on its designated children.

        Args:
            node (Node): the node to populate.
            depth (int): the current depth of this node.
            X (np.ndarray): the set of remaining training examples.
            y (np.ndarray): the set of remaining training labels.
        """
        if depth == self.max_depth or len(y) == 0:
            # set up this node as a majority classifier
            node.label = stats.mode(y)[0][0] if len(y) > 0 else 0.0
            print(f"    > assigning node LABEL {node.label}                     ", end="\r")
            return

        best_attribute, best_info = 0, np.inf
        for attribute in range(X.shape[1]):
            values = X[:, attribute]
            values = np.floor(values / (256/self.branch_factor))

            child_info = entropy(y, values)
            if child_info < best_info:
                best_attribute, best_info = attribute, child_info

        print(f"    > assigning node attribute {best_attribute}                     ", end="\r")
        node.attribute = best_attribute
        partition_values = X[:, best_attribute]
        partition_values = np.floor(partition_values / (256/self.branch_factor))

        for i in range(self.branch_factor):
            rows = np.where(partition_values==i)

            print(f"creating node at depth {depth+1}                     ", end="\r")
            child = Node(branch_factor=self.branch_factor)
            node.children.append(child)
            self.populate_node(child, depth + 1, X[rows], y[rows])
        
    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        print("creating root")
        self.root = Node(branch_factor=self.branch_factor)
        self.populate_node(self.root, 0, training_data, training_labels)

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        inferred_labels = []

        for i in tqdm(range(len(testing_data))):
            instance = testing_data[i]
            inferred_labels.append(self.root.route(instance))
        
        return inferred_labels
