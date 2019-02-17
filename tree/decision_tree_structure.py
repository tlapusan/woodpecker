import graphviz
import pygraphviz as pgv


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
        self.value = tree.tree_.value

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
                            label=f"Node {node_id} \n {self.features[self.feature[node_id]]} {threshold_sign} {self.threshold[node_id]} \n samples {self.n_node_samples[node_id]} \n value {self.value[node_id][0]}, \n impurity {round(self.impurity[node_id], 2)}",
                            fontsize=10, center=True, shape="ellipse")

            if self.children_left[node_id] != -1:
                g_tree.add_edge(node_id, self.children_left[node_id])
            if self.children_right[node_id] != -1:
                g_tree.add_edge(node_id, self.children_right[node_id])

        return graphviz.Source(g_tree.string())
