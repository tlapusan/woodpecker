import sys

import graphviz
import numpy as np
import pygraphviz as pgv
from matplotlib import pyplot as plt
from sklearn import tree as sklearn_tree


# TODO calculate max_depth (is useful when max_depth is not a parameter and we use for ex. min_samples_split = 100
# TODO add docs to each method
# TODO look at fast.ai random forest feature importance and other
# TODO ask opinions from other experience people in ML (george ciobanu, cristi lungu, cristi vicas)
# TODO try the same visualisation for different decision tree structures (ex. max_depth=[3, 5, 10, 20]


class DecisionTreeStructure:
    """A visual interpretation of decision tree structure.

    It contains two types of visualisations :
        - visualisations related to leaf nodes
        - visualisations about tree predictions

    Parameters
    ----------show_leaf_impurity
    tree : sklearn.tree.tree.DecisionTreeClassifier
        The tree to investigate

    features : list
        The list of features names

    Attributes
    ----------
    node_count : int
        The number of nodes from the tree

    children_left : array of int, shape[node_count]
        children_left[i] holds the node id of the left child node of node i.
        For leaves, children_left[i] == TREE_LEAF

    children_right : array of int, shape[node_count]
        children_right[i] holds the node id of the right child node of node i.
        For leaves, children_right[i] == TREE_LEAF

    feature : array of int, shape[node_count]
        feature[i] holds the feature index used for split at node i

    threshold : array of double, shape[node_count]
        threshold[i] holds the split threshold for node i

    impurity : array of double, shape[node_count]
        impurity[i] holds the impurity (ex. the value of splitting criterion) for node i

    n_node_samples : array of int, shape[node_count]
        n_node_samples[i] holds the number of training examples reaching the node i

    weighted_n_node_samples : array of int, shape[node_count]
        weighted_n_node_samples[i] holds the weighted number of training examples reaching the node i

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        value[i] holds the prediction value for node i

    is_leaf : array of bool, shape[node_count]
        is_leaf[i] holds true or false, depending if node i is a leaf or split node

    split_node_samples: array of int, shape[node_count]
        split_node_samples[i] holds training samples reaching the node i

    Methods
    -------
    show_features_importance()
        Show feature importance ordered by importance

    show_decision_tree_structure()
        Show decision tree structure as a binary tree.

    show_decision_tree_prediction_path(sample)
        Show only the decision path, from the whole tree, used for prediction.

    show_decision_tree_splits_prediction()
        Show the decision path for a specified sample, together with feature space splits

    show_leaf_impurity()
        Show only the leaf nodes associated with them impurity

    show_leaf_impurity_distribution()
        Show leaves impurities using a histogram

    show_leaf_samples()
        Show only the leaf nodes associated with them samples counts

    show_leaf_samples_by_class()
        Show only the leaf nodes associated with them samples counts, grouped by target class

    show_leaf_samples_distribution()
        Show leaves samples counts using a histogram




    """

    def __init__(self, tree, train_dataset, features, target):
        """Initialize necessary information about the tree.

        :param tree: sklearn.tree.tree.DecisionTreeClassifier
            The tree to investigate
        :param train_dataset: pandas.core.frame.DataFrame
            The training dataset the tree was trained on
        :param features: array of strings
            The list of features names used to train the tree
        :param target: str
            The name of target variable
        """

        self.tree = tree
        self.train_dataset = train_dataset
        self.features = features
        self.target = target

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
        self.split_node_samples = {}

    def show_decision_tree_structure(self):
        """Show decision tree structure as a binary tree.

        It is just an utility method for graphviz functionality to render a decision tree structure.

        :return: graphviz.files.Source
        """

        dot_data = sklearn_tree.export_graphviz(self.tree, out_file=None, feature_names=self.features,
                                                filled=True, rotate=True, node_ids=True)
        return graphviz.Source(dot_data)

    def show_features_importance(self, figsize=(20, 10)):
        """Visual representation of features importance.


        Features are ordered descending by their importance using a bar plot visualisation.
        oX contains features name and oY contains features importance.

        :param figsize: tuple
            the size (x, y) of the plot (default is (20, 10))
        :return: None
        """

        feature_names, feature_importances = zip(
            *sorted(list(zip(self.tree.feature_importances_, self.features)), key=lambda tup: tup[0],
                    reverse=True))
        plt.figure(figsize=figsize)
        plt.bar(feature_importances, feature_names)
        plt.xlabel("feature name", fontsize=20)
        plt.ylabel("feature importance", fontsize=20)
        plt.grid()
        plt.show()

    def show_decision_tree_prediction_path(self, sample):
        """Visual interpretation of prediction path.

        Show only the prediction path from a decision tree, instead of the whole tree.
        It helps to easily understand and follow the prediction path.
        The blue nodes are the nodes from prediction path and the black nodes are just blue nodes brothers.

        This kind of visualisation is very useful for debugging and understanding tree predictions.
        Also it is useful to explain to non technical people the reason behind tree predictions.

        :param sample: array of double, shape[features]
            The array of features values
        :return: graphviz.files.Source
        """

        node_indicator = self.tree.decision_path([sample])
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]
        g_tree = pgv.AGraph(strict=False, directed=True)
        g_tree.layout(prog='dot')

        for i in range(0, len(node_index)):

            node_id = node_index[i]

            if sample[self.feature[node_id]] <= self.threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # TODO round(self.value[node_id][0][0], 2) for regression tree
            g_tree.add_node(node_id, color="blue",
                            label=f"Node {node_id} \n {self.features[self.feature[node_id]]} {threshold_sign} {self.threshold[node_id]} \n samples {self.n_node_samples[node_id]} \n weighted sample {round(self.weighted_n_node_samples[node_id], 1)} \n values {self.value[node_id][0]}, \n impurity {round(self.impurity[node_id], 2)}",
                            fontsize=10, center=True, shape="ellipse")

            if self.children_left[node_id] != -1:
                g_tree.add_edge(node_id, self.children_left[node_id])

                if self.children_left[node_id] != node_index[i + 1]:
                    left_node_id = self.children_left[node_id]
                    g_tree.add_node(left_node_id,
                                    label=f"Node {left_node_id} \n feature split {self.features[self.feature[left_node_id]]} \n samples {self.n_node_samples[left_node_id]} \n values {self.value[left_node_id][0]}, \n impurity {round(self.impurity[left_node_id], 2)} ",
                                    fontsize=10, center=True, shape="ellipse")
            if self.children_right[node_id] != -1:
                g_tree.add_edge(node_id, self.children_right[node_id])
                if self.children_right[node_id] != node_index[i + 1]:
                    right_node_id = self.children_right[node_id]
                    g_tree.add_node(right_node_id,
                                    label=f"Node {right_node_id} \n feature split {self.features[self.feature[right_node_id]]} \n samples {self.n_node_samples[right_node_id]} \n values {self.value[right_node_id][0]}, \n impurity {round(self.impurity[right_node_id], 2)} ",
                                    fontsize=10, center=True, shape="ellipse")

        return graphviz.Source(g_tree.string())

    def _calculate_split_node_samples(self, dataset_training):
        decision_paths = self.tree.decision_path(dataset_training[self.features]).toarray()
        for index in dataset_training.index.values:
            decision_node_path = np.nonzero(decision_paths[index])[0]
            for node_id in decision_node_path:
                try:
                    self.split_node_samples[node_id].append(index)
                except KeyError as ex:
                    self.split_node_samples[node_id] = [index]

    # TODO add feature name for oX axe
    # TODO histogram do not reflect correctly the values size
    # TODO it is not clear now with transparency, make them on top ?
    def show_decision_tree_splits_prediction(self, sample, bins=10, figsize=(10, 5)):
        """Visual interpretation of features space splits for a specified sample.

        Show feature space splits for the tree nodes involved in prediction path for sample parameter.
        It is useful to

        :param sample: array of doubles, shape[features]
            The array of features values
        :return:
        """

        if len(self.split_node_samples) == 0:
            self._calculate_split_node_samples(self.train_dataset)

        print(list(zip(self.features, sample)))
        print()

        node_indicator = self.tree.decision_path([sample])
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]

        for node_id in node_index:

            # FIXME leaf node shows wrong information for feature
            # if self.feature[node_id] < 0:
            #   continue

            if (sample[self.feature[node_id]] <= self.threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            split_sample = self.train_dataset.iloc[self.split_node_samples[node_id]]
            print(
                f"nodeId {node_id}, {self.features[self.feature[node_id]]}({sample[self.feature[node_id]]}) {threshold_sign} {self.threshold[node_id]}, sample size {len(split_sample)}, impurity {round(self.impurity[node_id], 2)} ")
            #             split_sample.hist()
            print((len(split_sample.query(f"{self.target} == 0")), len(split_sample.query(f"{self.target} == 1"))))

            plt.figure(figsize=figsize)
            max_range = split_sample[self.features[self.feature[node_id]]].max()
            min_range = split_sample[self.features[self.feature[node_id]]].min()

            plt.hist(split_sample.query(f"{self.target} == 0")[self.features[self.feature[node_id]]],
                     label=f"{self.target} 0", bins=bins, range=(min_range, max_range))
            plt.hist(split_sample.query(f"{self.target} == 1")[self.features[self.feature[node_id]]],
                     alpha=0.8, label=f"{self.target} 1", bins=bins, range=(min_range, max_range))
            plt.axvline(self.threshold[node_id], c="red",
                        label=f"{self.features[self.feature[node_id]]} = {self.threshold[node_id]}")
            plt.xlabel(f"{self.features[self.feature[node_id]]} range of values", fontsize=14)
            plt.ylabel(f"training examples", fontsize=14)
            plt.legend()
            plt.show()

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
        plt.xlabel("leaf impurity", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    def show_leaf_impurity(self, figsize=None, show_type = "plot"):
        # TODO create a decorator
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        leaf_impurity = [(i, self.impurity[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        leaves, impurity = zip(*leaf_impurity)

        if show_type == "plot":
            if figsize:
                plt.figure(figsize=figsize)
            plt.xticks(range(0, len(leaves)), leaves)
            plt.bar(range(0, len(leaves)), impurity, label="leaf impurity")
            plt.xlabel("leaf node ids", fontsize=20)
            plt.ylabel("impurity", fontsize=20)
            plt.grid()
            plt.legend()
        elif show_type == "text":
            for leaf, impurity in leaf_impurity:
                print(leaf, impurity)

    def show_leaf_samples_distribution(self, bins=10, figsize=None, max_leaf_sample=sys.maxsize):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        if figsize:
            plt.figure(figsize=figsize)
        plt.hist([self.n_node_samples[i] for i in range(0, self.node_count) if
                  ((self.is_leaf[i]) & (self.n_node_samples[i] < max_leaf_sample))], bins=bins)
        plt.xlabel("leaf sample", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    def show_leaf_samples(self, figsize=None):
        if len(self.is_leaf) == 0:
            self._calculate_leaf_nodes()

        leaf_impurity = [(i, self.n_node_samples[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        x, y = zip(*leaf_impurity)

        if figsize:
            plt.figure(figsize=figsize)
        plt.xticks(range(0, len(x)), x)
        plt.bar(range(0, len(x)), y, label="leaf samples")
        plt.xlabel("leaf node ids", size=20)
        plt.ylabel("samples", size=20)
        plt.grid()
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
        plt.xlabel("leaf node ids", size=20)
        plt.ylabel("samples", size=20)
        plt.legend((p0[0], p1[0]), ('class 0 samples', 'class 1 samples'))
        # plt.show()


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
