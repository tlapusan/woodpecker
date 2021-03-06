import logging
import sys

import graphviz
import numpy as np
import pygraphviz as pgv
from matplotlib import pyplot as plt
from sklearn import tree as sklearn_tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# TODO change plot labels to display entropy or gini. and maybe from leaf to leaves ?
# TODO make more clear how to set show_leaf_predictions parameters

class DecisionTreeStructure:
    """A visual interpretation of decision tree structures.

    It contains two types of visualisations :
        - visualisations related to leaf nodes
        - visualisations about tree predictions

    Attributes
    ----------
    tree : sklearn.tree.DecisionTreeClassifier
        The tree to investigate

    train_dataset: pandas.core.frame.DataFrame
        The dataset the tree was trained on

    features: array of strings
        The list of features names used to train the tree

    target: str
        The name of target variable

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
    """

    def __init__(self, tree, train_dataset, features, target):
        """Initialize necessary information about the tree.

        :param tree: sklearn.tree.DecisionTreeClassifier
            The tree to investigate
        :param train_dataset: pandas.core.frame.DataFrame
            The training dataset the tree was trained on
        :param features: array of strings
            The list of features names used to train the tree
        :param target: str
            The name of target variable
        """

        self.tree = tree
        self.train_dataset = train_dataset.reset_index(drop=True)
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

    def _calculate_leaf_nodes(self):
        """Used to calculate the node type.

        The array is_leaf[index] will be True in case the node with id=index is a leaf,
        or False if the node is a split node.
        """

        if len(self.is_leaf) == 0:
            self.is_leaf = np.zeros(shape=self.node_count, dtype=bool)
            stack = [(0)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id = stack.pop()
                # If we have a test node
                if self.children_left[node_id] != self.children_right[node_id]:
                    stack.append((self.children_left[node_id]))
                    stack.append((self.children_right[node_id]))
                else:
                    self.is_leaf[node_id] = True

    def show_decision_tree_structure(self, rotate=True):
        """Show decision tree structure as a binary tree.

        It is just an utility method for graphviz functionality to render a decision tree structure.

        :return: graphviz.files.Source
        """

        dot_data = sklearn_tree.export_graphviz(self.tree, out_file=None, feature_names=self.features,
                                                filled=True, rotate=rotate, node_ids=True)
        return graphviz.Source(dot_data)

    def show_features_importance(self, barh=False, max_feature_to_display=None, figsize=(20, 10)):
        """Visual representation of features importance for DecisionTree.


        Features are ordered descending by their importance using a bar plot visualisation.

        :param max_feature_to_display: int
            Maximum number of features to display. This is useful in case we have hundreds of features and the
            plot become incomprehensible.
        :param barh: boolean
            True if we want to display feature importance into a bath plot, false otherwise
        :param figsize: tuple
            the size (x, y) of the plot (default is (20, 10))
        :return: None
        """

        feature_importances, feature_names = zip(
            *sorted(list(zip(self.tree.feature_importances_, self.features)), key=lambda tup: tup[0],
                    reverse=True))

        if max_feature_to_display is not None:
            feature_names = feature_names[:max_feature_to_display]
            feature_importances = feature_importances[:max_feature_to_display]

        plt.figure(figsize=figsize)
        if barh:
            plt.barh(feature_names, feature_importances)
        else:
            plt.bar(feature_names, feature_importances)

        plt.xlabel("feature name", fontsize=20)
        plt.ylabel("feature importance", fontsize=20)
        plt.grid()
        plt.show()

    def _get_node_path_info(self, node_id, sample, is_weighted):
        self._calculate_leaf_nodes()

        sample_value = round(sample[self.feature[node_id]], 2)

        if sample_value <= self.threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        newline = "\n"

        if isinstance(self.tree, DecisionTreeClassifier):
            # I created bellow local variables because of reformat code issues(when the whole statement is in return)
            split_value = self.features[self.feature[node_id]] + '(' + str(
                sample[self.feature[node_id]]) + ') ' + threshold_sign + ' ' + str(
                round(self.threshold[node_id], 2)) + newline if not self.is_leaf[node_id] else ''
            weighted_sample_value = 'weighted sample ' + str(
                round(self.weighted_n_node_samples[node_id], 1)) + newline if is_weighted else ''
            return f"Node {node_id} \n" \
                f"{split_value}" \
                f"samples {self.n_node_samples[node_id]} \n" \
                f"{weighted_sample_value}" \
                f"values {self.value[node_id][0]}, \n" \
                f"impurity {round(self.impurity[node_id], 2)}"
        elif isinstance(self.tree, DecisionTreeRegressor):
            split_value = self.features[self.feature[node_id]] + '(' + str(
                sample[self.feature[node_id]]) + ') ' + threshold_sign + ' ' + str(
                round(self.threshold[node_id], 2)) + newline if not self.is_leaf[node_id] else ''
            weighted_sample_value = 'weighted sample ' + str(
                round(self.weighted_n_node_samples[node_id], 1)) + newline if is_weighted else ''
            return f"Node {node_id} \n" \
                f"{split_value}" \
                f"samples {self.n_node_samples[node_id]} \n" \
                f"{weighted_sample_value}" \
                f"prediction {self.value[node_id][0][0]}, \n" \
                f"{self.tree.criterion.upper()} {round(self.impurity[node_id], 2)}"

    def show_decision_tree_prediction_path(self, sample, is_weighted=False):
        """Visual interpretation of prediction path.

        Show only the prediction path from a decision tree, instead of the whole tree.
        It helps to easily understand and follow the prediction path.
        The blue nodes are the nodes from prediction path and the black nodes are just blue nodes brothers.

        This kind of visualisation is very useful for debugging and understanding tree predictions.
        Also it is useful to explain to non technical people the reason behind tree predictions.

        :param is_weighted: boolean
            Whether or not to include weighted number of training samples reaching node i.
        :param sample: array of double, shape[features]
            The array of features values
        :return: graphviz.files.Source
        """

        logging.info(f"Make a prediction for sample {sample}")

        node_indicator = self.tree.decision_path([sample])
        decision_node_path = node_indicator.indices[node_indicator.indptr[0]:
                                                    node_indicator.indptr[1]]
        logging.info(f"decision path {decision_node_path}")

        g_tree = pgv.AGraph(strict=False, directed=True)
        g_tree.layout(prog='dot')
        for i in range(0, len(decision_node_path)):
            node_id = decision_node_path[i]
            node_label = self._get_node_path_info(node_id, sample, is_weighted)
            logging.debug(f"adding node id {node_id} with label {node_label}")

            g_tree.add_node(node_id, color="blue", label=node_label, fontsize=10, center=True, shape="ellipse")

            # check if node_id is not a leaf
            if self.children_left[node_id] != -1:
                g_tree.add_edge(node_id, self.children_left[node_id])

                # check if children_left[node_id] is not from the path and plot the node with black (neighbor node)
                if self.children_left[node_id] != decision_node_path[i + 1]:
                    left_node_id = self.children_left[node_id]
                    g_tree.add_node(left_node_id, label=self._get_node_path_info(left_node_id, sample, is_weighted),
                                    fontsize=10,
                                    center=True, shape="ellipse")

            # check if node_id is not a leaf
            if self.children_right[node_id] != -1:
                g_tree.add_edge(node_id, self.children_right[node_id])

                # check if children_right[node_id] is not from the path and plot the node with black (neighbor node)
                if self.children_right[node_id] != decision_node_path[i + 1]:
                    right_node_id = self.children_right[node_id]
                    g_tree.add_node(right_node_id, label=self._get_node_path_info(right_node_id, sample, is_weighted),
                                    fontsize=10,
                                    center=True, shape="ellipse")

        return graphviz.Source(g_tree.string())

    def _calculate_split_node_samples(self, dataset_training):
        decision_paths = self.tree.decision_path(dataset_training[self.features]).toarray()
        logging.info(f"decision paths {decision_paths} ")
        for index in dataset_training.index.values:
            decision_node_path = np.nonzero(decision_paths[index])[0]
            for node_id in decision_node_path:
                try:
                    self.split_node_samples[node_id].append(index)
                except KeyError as ex:
                    self.split_node_samples[node_id] = [index]

    def get_node_samples(self, node_id, return_type="plain"):
        """Create a dataframe containing all training samples reaching node_id.

        :param return_type: str
            Specify different types of outputs:
             'plain' to return all samples from the training set as a dataframe
             'describe' to generate statistics that summarize the samples
             'describe_by_class' to generate statistics that summarize th samples, but by class target variable
        :param node_id: int
            The id of node_id
        :return: pandas.DataFrame
        """

        if len(self.split_node_samples) == 0:
            self._calculate_split_node_samples(self.train_dataset)

        # print(self.split_node_samples)
        output = self.train_dataset.iloc[self.split_node_samples[node_id]][self.features + [self.target]]. \
            sort_values(by=self.target)

        if return_type == "plain":
            return output
        elif return_type == "describe":
            return output.describe()
        elif return_type == "describe_by_class":
            return output.groupby(self.target).describe().transpose()

    def show_leaf_samples_distribution(self, min_samples=0, max_samples=sys.maxsize, bins=10, figsize=None):
        """ Visualize distribution of leaves samples.

        :param bins: int
            Number of bins of histograms
        :param figsize: tuple of int
            The figure size to be displayed
        """

        self._calculate_leaf_nodes()
        if figsize:
            plt.figure(figsize=figsize)
        plt.hist([self.n_node_samples[i] for i in range(0, self.node_count) if
                  self.is_leaf[i] and self.n_node_samples[i] >= min_samples and self.n_node_samples[i] <= max_samples],
                 bins=bins)
        plt.xlabel("leaf sample", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    def show_leaf_samples(self, figsize=(10, 5), display_type="plot"):
        """Show number of training samples from each leaf.

        If display_type = 'plot' it will show leaves samples using a plot.
        If display_type = 'text' it will show leaves samples as text. This method is preferred if number
        of leaves is very large and we cannot determine clearly the leaves from the plot.

        :param figsize: tuple of int
            The plot size
        :param display_type: str, optional
            'plot' or 'text'
        """

        self._calculate_leaf_nodes()
        leaf_samples = [(i, self.n_node_samples[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        x, y = zip(*leaf_samples)

        if display_type == "plot":
            if figsize:
                plt.figure(figsize=figsize)
            plt.xticks(range(0, len(x)), x)
            plt.bar(range(0, len(x)), y)
            plt.xlabel("leaf node ids", size=20)
            plt.ylabel("samples count", size=20)
            plt.grid()
        elif display_type == "text":
            for leaf, samples in leaf_samples:
                print(leaf, samples)

    def get_leaf_node_count(self):
        """Get number of leaves from the tree

        :return: int
            Number of leaves
        """

        self._calculate_leaf_nodes()
        return sum(self.is_leaf)

    def get_split_node_count(self):
        """Get number of split nodes from the tree

        :return: int
            Number of split nodes
        """

        self._calculate_leaf_nodes()
        return len(self.is_leaf) - sum(self.is_leaf)

    def get_node_count(self):
        """Get total number of nodes from the tree

        :return: int
            Total number of nodes
        """

        return self.node_count
