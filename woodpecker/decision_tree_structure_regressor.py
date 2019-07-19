import logging

import graphviz
import pygraphviz as pgv
import matplotlib.pyplot as plt

from woodpecker.decision_tree_structure import DecisionTreeStructure

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class DecisionTreeStructureRegressor(DecisionTreeStructure):

    def show_leaf_regression_criteria(self, figsize=(10, 5), display_type="plot"):
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
        self._calculate_leaf_nodes()

        plt.figure(figsize=figsize)
        plt.hist([self.impurity[i] for i in range(0, self.node_count) if self.is_leaf[i]], bins=bins)
        plt.xlabel(f"leaf {self.tree.criterion.upper()}", fontsize=20)
        plt.ylabel("leaf count", fontsize=20)