import logging

from matplotlib import pyplot as plt

from woodpecker.sklearn.decisiontree.decision_tree_structure import DecisionTreeStructure

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class DecisionTreeStructureClassifier(DecisionTreeStructure):
    """A class used to visually interpret the DecisionTreeClassifier structure from scikit-learn library.

    Parent class, DecisionTreeStructure, contains the common visualisation methods which can be used for both
    classification and regression trees.
    """

    def show_leaf_impurity(self, figsize=None, display_type="plot"):
        """Show impurity for each leaf.

        If display_type = 'plot' it will show leaves impurities using a plot.
        If display_type = 'text' it will show leaves impurities as text. This method is preferred if number
        of leaves is very large and the plot became very big and hard to interpret.

        :param figsize: tuple of int
            The plot size
        :param display_type: str, optional
            'plot' or 'text'
        """

        self._calculate_leaf_nodes()
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

    def show_leaf_impurity_distribution(self, bins=10, figsize=(10, 5)):
        """Visualize distribution of leaves impurities.

        :param bins: int
            Number of bins of histograms
        :param figsize: tuple of int
            The figure size to be displayed
        """

        self._calculate_leaf_nodes()

        if figsize:
            plt.figure(figsize=figsize)
        plt.hist([self.impurity[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)
        plt.xlabel("leaf impurity", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)

    # TODO make it more explicit
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

        self._calculate_leaf_nodes()
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

    # TODO it is not clear now with transparency, make them on top ?
    def show_decision_tree_splits_prediction(self, sample, bins=10, figsize=(10, 5)):
        """Visual interpretation of features space splits for a specified sample prediction.

        Show feature space splits for the tree nodes involved in prediction path for sample parameter.
        It is useful to

        :param figsize: tuple of int
            The figure size to be displayed
        :param bins: int
            Number of bins from histogram
        :param sample: array of doubles, shape[features]
            The array of features values
        """

        self._calculate_leaf_nodes()

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
            sample_size = f"sample size {len(split_sample)}"
            class_0_size = len(split_sample.query(f'{self.target} == 0'))
            class_1_size = len(split_sample.query(f'{self.target} == 1'))
            sample_size_distribution = f"({class_0_size}/{class_1_size})"

            if self.is_leaf[node_id]:
                plt.title(
                    f"Node {node_id}, "
                    f"{sample_size}{sample_size_distribution}, "
                    f"impurity {round(self.impurity[node_id], 2)}")
            else:
                feature_split_name = f"{self.features[self.feature[node_id]]}({sample[self.feature[node_id]]})"
                plt.title(
                    f"Node {node_id}, "
                    f"{feature_split_name} {threshold_sign} {self.threshold[node_id]}, "
                    f"{sample_size}{sample_size_distribution}, "
                    f"impurity {round(self.impurity[node_id], 2)}")

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


def show_leaf_samples_by_class(self, figsize=None, leaf_sample_size=None, plot_ylim=None):
    """Show samples by class for each leaf.

    :param plot_ylim: int, optional
        The max value for oY. This is useful in case we have few leaves with big sample values which 'shadow'
        the other leaves values
    :param leaf_sample_size: int, optional
        The sample of leaves to plot. This is useful when the tree contains to many leaves and cannot be displayed
        clear in a plot.
    :param figsize: tuple of int, optional
        The plot size
    """

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

    if plot_ylim is not None:
        plt.ylim(0, plot_ylim)

    plt.xlabel("leaf node ids", size=20)
    plt.ylabel("samples", size=20)
    plt.grid()
    plt.legend((p0[0], p1[0]), ('class 0 samples', 'class 1 samples'))
