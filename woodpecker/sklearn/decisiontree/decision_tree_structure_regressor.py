import logging

import matplotlib.pyplot as plt

from woodpecker.sklearn.decisiontree.decision_tree_structure import DecisionTreeStructure

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class DecisionTreeStructureRegressor(DecisionTreeStructure):
    """A class used to visually interpret the DecisionTreeRegressor structure from scikit-learn library.

    Most of the functionalities are implemented in the parent class. Visit DecisionTreeStructure to find out more !
    """

    def show_leaf_regression_criteria(self, figsize=(10, 5), display_type="plot"):
        """Visualize leaf regression criterias for each leaf.

        Main criterias for regression trees are MAE and MSE.
        Visualising them, it is easier to see each leaf performance. You can visualize the criterias in both plain text
        and plots.

        :param figsize: tuple of int, optional
            The plot size
        :param display_type: str, optional
            Display type can be text or plot. (default plot)
        """

        self._calculate_leaf_nodes()
        leaf_impurity = [(i, self.impurity[i]) for i in range(0, self.node_count) if self.is_leaf[i]]
        leaves, impurity = zip(*leaf_impurity)

        if display_type == "plot":
            if figsize:
                plt.figure(figsize=figsize)
            plt.xticks(range(0, len(leaves)), leaves)
            plt.bar(range(0, len(leaves)), impurity)
            plt.xlabel("leaf node ids", fontsize=20)
            plt.ylabel(f"leaf {self.tree.criterion.upper()}", fontsize=20)
            plt.grid()
            # plt.legend()
        elif display_type == "text":
            for leaf, impurity in leaf_impurity:
                print(leaf, impurity)

    def show_leaf_regression_criteria_distribution(self, bins=10, figsize=(10, 5)):
        """ Visualize distribution of leaves criterias.

        It is useful when you want the see the overall leaf performances from the tree.

        :param bins: int, optional
            Number of bins from histogram (default 10)
        :param figsize: tuple, optional
            The plot size.
        """

        self._calculate_leaf_nodes()

        plt.figure(figsize=figsize)
        plt.hist([self.impurity[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)
        plt.xlabel(f"leaf {self.tree.criterion.upper()}", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)
