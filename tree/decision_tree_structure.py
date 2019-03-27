import sys

import graphviz
import numpy as np
import pygraphviz as pgv
from matplotlib import pyplot as plt


# TODO calculate max_depth (is useful when max_depth is not a parameter and we use for ex. min_samples_split = 100

class DecisionTreeStructure:
    def __init__(self, tree, features):
        self.tree = tree
        self.features = features

        self.node_count = tree.tree_.node_count
        self.children_left = tree.tree_.children_left
        self.children_right = tree.tree_.children_right
        self.feature = tree.tree_.feature
        self.threshold = tree.tree_.threshold
        self.impurity = tree.tree_.impurity
        self.n_node_samples = tree.tree_.n_node_samples
        self.weighted_n_node_samples = tree.tree_.weighted_n_node_samples
        self.value = tree.tree_.value

        self.is_leaf = []

    def show_decision_tree_prediction_path(self, sample):
        node_indicator = self.tree.decision_path([sample])
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]
        g_tree = pgv.AGraph(strict=False, directed=True)
        g_tree.layout(prog='dot')

        for node_id in node_index:
            #     if leave_id[sample_id] == node_id:
            #         continue

            if (sample[self.feature[node_id]] <= self.threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # TODO round(self.value[node_id][0][0], 2) for regression tree
            g_tree.add_node(node_id, color="blue",
                            label=f"Node {node_id} \n {self.features[self.feature[node_id]]} {threshold_sign} {self.threshold[node_id]} \n samples {self.n_node_samples[node_id]} \n weighted sample {round(self.weighted_n_node_samples[node_id], 1)} \n value {self.value[node_id][0]}, \n impurity {round(self.impurity[node_id], 2)}",
                            fontsize=10, center=True, shape="ellipse")

            if self.children_left[node_id] != -1:
                g_tree.add_edge(node_id, self.children_left[node_id])
            if self.children_right[node_id] != -1:
                g_tree.add_edge(node_id, self.children_right[node_id])

        return graphviz.Source(g_tree.string())

    def _calculate_leaf_nodes(self):
        self.is_leaf = np.zeros(shape=self.node_count, dtype=bool)
        stack = [(0)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id = stack.pop()
            # If we have a test node
            if (self.children_left[node_id] != self.children_right[node_id]):
                stack.append((self.children_left[node_id]))
                stack.append((self.children_right[node_id]))
            else:
                self.is_leaf[node_id] = True

    def show_leaf_impurity_distribution(self, bins=10, figsize=None):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        if figsize:
            plt.figure(figsize=figsize)
        plt.xticks(np.arange(0.0, 1.0, 0.05))
        plt.hist([self.impurity[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)

    def show_leaf_impurity(self, figsize=None):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        leaf_impurity = [(i, self.impurity[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        x, y = zip(*leaf_impurity)

        if figsize:
            plt.figure(figsize=figsize)
        plt.xticks(range(0, len(x)), x)
        plt.bar(range(0, len(x)), y, label="leaf impurity")
        plt.legend()

    def show_leaf_samples_distribution(self, bins=10, figsize=None, max_leaf_sample=sys.maxsize):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        if figsize:
            plt.figure(figsize=figsize)
        plt.hist([self.n_node_samples[i] for i in range(0, self.node_count) if
                  ((self.is_leaf[i]) & (self.n_node_samples[i] < max_leaf_sample))], bins=bins)

    def show_leaf_samples(self, figsize=None):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        leaf_impurity = [(i, self.n_node_samples[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        x, y = zip(*leaf_impurity)

        if figsize:
            plt.figure(figsize=figsize)
        plt.xticks(range(0, len(x)), x)
        plt.bar(range(0, len(x)), y, label="leaf samples")
        plt.legend()

    def show_leaf_samples_by_class(self, figsize=None, leaf_sample_size=None):
        """
        For now only for binary classification
        """
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        leaf_samples = [(i, self.value[i][0][0], self.value[i][0][1], self.impurity[i]) for i in
                        range(0, self.node_count)
                        if (self.is_leaf[i])]
        index, leaf_samples_0, leaf_samples_1, impurity_sample = zip(*leaf_samples)

        if leaf_sample_size is None:
            leaf_sample_size = len(index)
        if figsize:
            plt.figure(figsize=figsize)
        p0 = plt.bar(range(0, len(index[:leaf_sample_size])), leaf_samples_0[:leaf_sample_size])
        p1 = plt.bar(range(0, len(index[:leaf_sample_size])), leaf_samples_1[:leaf_sample_size],
                     bottom=leaf_samples_0[:leaf_sample_size])
        plt.xticks(range(0, len(index)), index)
        plt.legend((p0[0], p1[0]), ('class 0', 'class 1'))
        plt.show()

    def show_features_importance(self, figsize=(20, 10)):
        feature_importances = dict(
            sorted(list(zip(self.tree.feature_importances_, self.features)), key=lambda tup: tup[0],
                   reverse=True))
        plt.figure(figsize=figsize)
        plt.bar(dict(feature_importances).values(), dict(feature_importances).keys())
        plt.grid()
        plt.show()

    def get_leaf_node_count(self):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        return sum(self.is_leaf)

    def get_split_node_count(self):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        return len(self.is_leaf) - sum(self.is_leaf)

    def get_node_count(self):
        return self.node_count
