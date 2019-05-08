import logging

import graphviz
import numpy as np
import pygraphviz as pgv
from matplotlib import pyplot as plt
from sklearn import tree as sklearn_tree

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# TODO change plot labels to display entropy or gini. and maybe from leaf to leaves ?
# TODO make more clear how to set show_leaf_predictions parameters

class DecisionTreeStructure:
    """A visual interpretation of decision woodpecker structure. Only for classification for the moment.

    It contains two types of visualisations :
        - visualisations related to leaf nodes
        - visualisations about woodpecker predictions

    Parameters
    ----------show_leaf_impurity
    woodpecker : sklearn.woodpecker.woodpecker.DecisionTreeClassifier
        The woodpecker to investigate

    features : list
        The list of features names

    Attributes
    ----------
    node_count : int
        The number of nodes from the woodpecker

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
        Show decision woodpecker structure as a binary woodpecker.

    show_decision_tree_prediction_path(sample)
        Show only the decision path, from the whole woodpecker, used for prediction.

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

    show_leaf_predictions()
        Show number of correct/wrong predictions for each leaf

    """

    def __init__(self, tree, train_dataset, features, target):
        """Initialize necessary information about the woodpecker.

        :param tree: sklearn.woodpecker.woodpecker.DecisionTreeClassifier
            The woodpecker to investigate
        :param train_dataset: pandas.core.frame.DataFrame
            The training dataset the woodpecker was trained on
        :param features: array of strings
            The list of features names used to train the woodpecker
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

    def _calculate_leaf_nodes(func):
        """Decorator used to calculate the node type.

        The array is_leaf[index] will be True in case the node with id=index is a leaf,
        or False if the node is a split node.
        """

        def wrapper(self, *args, **kwargs):
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

            return func(self, *args, **kwargs)

        return wrapper

    def show_decision_tree_structure(self, rotate=True):
        """Show decision woodpecker structure as a binary woodpecker.

        It is just an utility method for graphviz functionality to render a decision woodpecker structure.

        :return: graphviz.files.Source
        """

        dot_data = sklearn_tree.export_graphviz(self.tree, out_file=None, feature_names=self.features,
                                                filled=True, rotate=rotate, node_ids=True)
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

    @_calculate_leaf_nodes
    def _get_node_path_info(self, node_id, sample, is_weighted):
        sample_value = round(sample[self.feature[node_id]], 2)

        if sample_value <= self.threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        newline = "\n"
        return f"Node {node_id} \n" \
               f"{self.features[self.feature[node_id]] + '(' + str(sample[self.feature[node_id]]) + ') ' +  threshold_sign + ' ' + str(round(self.threshold[node_id], 2)) + newline if not self.is_leaf[node_id] else ''}" \
               f" samples {self.n_node_samples[node_id]} \n" \
               f" {'weighted sample ' + str(round(self.weighted_n_node_samples[node_id], 1)) + newline if is_weighted else ''}" \
               f"values {self.value[node_id][0]}, \n " \
               f"impurity {round(self.impurity[node_id], 2)}"

    def show_decision_tree_prediction_path(self, sample, is_weighted=False):
        """Visual interpretation of prediction path.

        Show only the prediction path from a decision woodpecker, instead of the whole woodpecker.
        It helps to easily understand and follow the prediction path.
        The blue nodes are the nodes from prediction path and the black nodes are just blue nodes brothers.

        This kind of visualisation is very useful for debugging and understanding woodpecker predictions.
        Also it is useful to explain to non technical people the reason behind woodpecker predictions.

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

    def get_node_samples(self, node_id):
        """Create a dataframe containing all training samples reaching node_id.

        :param node_id: int
            The id of node_id
        :return: pandas.DataFrame
        """

        if len(self.split_node_samples) == 0:
            self._calculate_split_node_samples(self.train_dataset)

        return self.train_dataset.iloc[self.split_node_samples[node_id]]

    # TODO it is not clear now with transparency, make them on top ?
    @_calculate_leaf_nodes
    def show_decision_tree_splits_prediction(self, sample, bins=10, figsize=(10, 5)):
        """Visual interpretation of features space splits for a specified sample.

        Show feature space splits for the woodpecker nodes involved in prediction path for sample parameter.
        It is useful to

        :param figsize: tuple of int
            The figure size to be displayed
        :param bins: int
            Number of bins from histogram
        :param sample: array of doubles, shape[features]
            The array of features values
        """

        if len(self.split_node_samples) == 0:
            self._calculate_split_node_samples(self.train_dataset)

        print(list(zip(self.features, sample)))
        print()

        node_indicator = self.tree.decision_path([sample])
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[1]]

        for node_id in node_index:
            if sample[self.feature[node_id]] <= self.threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            split_sample = self.train_dataset.iloc[self.split_node_samples[node_id]]

            plt.figure(figsize=figsize)
            if self.is_leaf[node_id]:
                plt.title(
                    f"Node {node_id}, sample size {len(split_sample)} ({len(split_sample.query(f'{self.target} == 0'))}/{len(split_sample.query(f'{self.target} == 1'))}), impurity {round(self.impurity[node_id], 2)} ")
            else:
                plt.title(
                    f"Node {node_id}, {self.features[self.feature[node_id]]}({sample[self.feature[node_id]]}) {threshold_sign} {self.threshold[node_id]}, sample size {len(split_sample)} ({len(split_sample.query(f'{self.target} == 0'))}/{len(split_sample.query(f'{self.target} == 1'))}), impurity {round(self.impurity[node_id], 2)} ")
            max_range = split_sample[self.features[self.feature[node_id]]].max()
            min_range = split_sample[self.features[self.feature[node_id]]].min()

            plt.hist(split_sample.query(f"{self.target} == 0")[self.features[self.feature[node_id]]],
                     label=f"{self.target} 0", bins=bins, range=(min_range, max_range))
            plt.hist(split_sample.query(f"{self.target} == 1")[self.features[self.feature[node_id]]],
                     alpha=0.8, label=f"{self.target} 1", bins=bins, range=(min_range, max_range))

            if not self.is_leaf[node_id]:
                plt.axvline(self.threshold[node_id], c="red",
                            label=f"{self.features[self.feature[node_id]]} = {self.threshold[node_id]}")

            plt.xlabel(f"{self.features[self.feature[node_id]]} range of values", fontsize=14)
            plt.ylabel(f"node examples", fontsize=14)
            plt.legend()
            plt.show()

    @_calculate_leaf_nodes
    def show_leaf_impurity_distribution(self, bins=10, figsize=None):
        """ Visualize distribution of leaves impurities

        :param bins: int
            Number of bins of histograms
        :param figsize: tuple of int
            The figure size to be displayed
        """

        if figsize:
            plt.figure(figsize=figsize)
        plt.xticks(np.arange(0.0, 1.0, 0.05))
        plt.hist([self.impurity[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)
        plt.xlabel("leaf impurity", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    @_calculate_leaf_nodes
    def show_leaf_impurity(self, figsize=None, display_type="plot"):
        """Show impurity for each leaf.

        If display_type = 'plot' it will show leaves impurities using a plot.
        If display_type = 'text' it will show leaves impurities as text. This method is preferred if number
        of leaves is very large and we cannot determine clearly the leaves from the plot.

        :param figsize: tuple of int
            The plot size
        :param display_type: str, optional
            'plot' or 'text'
        """

        leaf_impurity = [(i, self.impurity[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        leaves, impurity = zip(*leaf_impurity)

        if display_type == "plot":
            if figsize:
                plt.figure(figsize=figsize)
            plt.xticks(range(0, len(leaves)), leaves)
            plt.bar(range(0, len(leaves)), impurity, label="leaf impurity")
            plt.xlabel("leaf node ids", fontsize=20)
            plt.ylabel("impurity", fontsize=20)
            plt.grid()
            plt.legend()
        elif display_type == "text":
            for leaf, impurity in leaf_impurity:
                print(leaf, impurity)

    @_calculate_leaf_nodes
    def show_leaf_samples_distribution(self, bins=10, figsize=None):
        """ Visualize distribution of leaves samples.

        :param bins: int
            Number of bins of histograms
        :param figsize: tuple of int
            The figure size to be displayed
        """

        if figsize:
            plt.figure(figsize=figsize)
        plt.hist([self.n_node_samples[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)
        plt.xlabel("leaf sample", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    @_calculate_leaf_nodes
    def show_leaf_samples(self, figsize=None, display_type="plot"):
        """Show samples for each leaf.

        If display_type = 'plot' it will show leaves samples using a plot.
        If display_type = 'text' it will show leaves samples as text. This method is preferred if number
        of leaves is very large and we cannot determine clearly the leaves from the plot.

        :param figsize: tuple of int
            The plot size
        :param display_type: str, optional
            'plot' or 'text'
        """

        leaf_samples = [(i, self.n_node_samples[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        x, y = zip(*leaf_samples)

        if display_type == "plot":
            if figsize:
                plt.figure(figsize=figsize)
            plt.xticks(range(0, len(x)), x)
            plt.bar(range(0, len(x)), y, label="leaf samples")
            plt.xlabel("leaf node ids", size=20)
            plt.ylabel("samples", size=20)
            plt.grid()
            plt.legend()
        elif display_type == "text":
            for leaf, samples in leaf_samples:
                print(leaf, samples)

    @_calculate_leaf_nodes
    def show_leaf_samples_by_class(self, figsize=None, leaf_sample_size=None):
        """Show samples by class for each leaf.
        
        :param figsize: tuple of int
            The plot size
        """

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
        plt.grid()
        plt.legend((p0[0], p1[0]), ('class 0 samples', 'class 1 samples'))

    @_calculate_leaf_nodes
    def get_leaf_node_count(self):
        """Get number of leaves from the woodpecker
        
        :return: int
            Number of leaves
        """

        return sum(self.is_leaf)

    @_calculate_leaf_nodes
    def get_split_node_count(self):
        """Get number of split nodes from the woodpecker
        
        :return: int
            Number of split nodes 
        """

        return len(self.is_leaf) - sum(self.is_leaf)

    def get_node_count(self):
        """Get total number of nodes from the woodpecker
        
        :return: int
            Total number of nodes
        """

        return self.node_count

    @_calculate_leaf_nodes
    def show_leaf_predictions(self, dataset, target, figsize=(20, 7)):
        """Show number of correct/wrong predictions for each leaf.

        It's useful for :
            - to see which leaves are participating for dataset predictions
            - to see leaves performance for the dataset

        :param dataset: pandas.DataFrame
            Dataset for which we will make predictions
        :param target: list
            True targets
        :param figsize: tuple of int
            The plot size
        """

        x_predictions = list(self.tree.predict(dataset[self.features]))
        prediction_correct = [0] * len(dataset)
        prediction_wrong = [0] * len(dataset)

        node_indicator = self.tree.decision_path(dataset[self.features])
        for i in range(len(dataset)):
            prediction_path = node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i + 1]]
            prediction_leaf = prediction_path[len(prediction_path) - 1]

            if x_predictions[i] == target[i]:
                prediction_correct[prediction_leaf] += 1
            else:
                prediction_wrong[prediction_leaf] += 1

        prediction_correct = [prediction_correct[i] for i in range(len(self.is_leaf)) if self.is_leaf[i]]
        prediction_wrong = [prediction_wrong[i] for i in range(len(self.is_leaf)) if self.is_leaf[i]]
        leaf_indices = [i for i in range(len(self.is_leaf)) if self.is_leaf[i]]

        plt.figure(figsize=figsize)
        plt.xticks(range(len(prediction_correct)), leaf_indices)
        plt.bar(range(len(prediction_correct)), prediction_correct, label="correct predictions3")
        plt.bar(range(len(prediction_wrong)), prediction_wrong, bottom=prediction_correct, label="wrong predictions")
        plt.xlabel("leaf node ids", fontsize=20)
        plt.legend()
