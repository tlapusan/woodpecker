import statistics

import matplotlib.pyplot as plt


class RandomForestStructureClassifier:
    def __init__(self, rf, train_dataset, features, target):
        self.rf = rf
        self.train_dataset = train_dataset
        self.features = features
        self.target = target

    def show_features_importance(self, barh=False, max_feature_to_display=None, figsize=(20, 10)):
        """Visual representation of features importance for RandomForest.

        Features are ordered descending by their importance using a bar plot visualisation.

        :param max_feature_to_display: int
            Maximum number of features to display. This is useful in case we have hundreds of features and the
            plot become incomprehensible.
        :param barh: boolean
            True if we want to display feature importances into a bath plot, false otherwise
        :param figsize: tuple
            the size (x, y) of the plot (default is (20, 10))
        :return: None
        """

        feature_importances, feature_names = zip(
            *sorted(list(zip(self.rf.feature_importances_, self.features)), key=lambda tup: tup[0],
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

    # TODO here I have dataset as method parameter and others, like features, are instance arguments.
    # TODO Should I make them all method parameters ?
    def show_prediction_probabilities_distribution(self, dataset, correct_prediction=True, by_class=False,
                                                   figsize=(20, 10)):
        """Visualize prediction probabilities for correct or incorrect predictions.

        This is useful to see model confidence for its predictions. Predicted class is chosen based on it's prediction
        probabilities. A class is predicted if it has a probability between 0.5 and 1.0. Having a higher probability,
        it means that the model is more confident for predicted class.


        :param by_class: boolean
            by_class=True if we want to see prediction distribution by classes,
            by_class=False if we want to see prediction distribution in general
        :param dataset: pandas dataframe
            The dataset to make predictions on.
        :param correct_prediction: boolean
            correct_prediction=True if we want to see correct predicted classes, False otherwise.
        :param figsize: tuple
            The plot size
        """

        dataset = dataset.copy()
        dataset["predicted_class"] = self.rf.predict(dataset[self.features])
        dataset["predicted_max_proba"] = [max(predict_proba) for predict_proba in
                                          self.rf.predict_proba(dataset[self.features])]
        if correct_prediction:
            dataset = dataset[dataset[self.target] == dataset["predicted_class"]]
        elif not correct_prediction:
            dataset = dataset[dataset[self.target] != dataset["predicted_class"]]

        _ = plt.figure(figsize=figsize)
        if not by_class:
            _ = dataset.predicted_max_proba.plot.hist(bins=30)
        elif by_class:
            _ = dataset[dataset[self.target] == 0].predicted_max_proba.plot.hist(bins=30, label="class 0")
            _ = dataset[dataset[self.target] == 1].predicted_max_proba.plot.hist(bins=30, label="class 1")
            _ = plt.legend()

    # TODO find a better method name
    # TODO add docs
    def show_prediction_std(self, dataset, figsize=(20, 10), bins=20):
        prediction_std = []
        prediction_mean = []
        for sample in dataset[self.features].iterrows():
            # TODO there are cases when predicted_class is not 0 or 1. It could be strings for example. Treat this cases.
            predicted_class = self.rf.predict([sample[1]])[0]
            predictions = [estimator.predict_proba([sample[1]])[0][predicted_class] for estimator in
                           self.rf.estimators_]
            prediction_mean.append(statistics.mean(predictions))
            prediction_std.append(statistics.stdev(predictions))

        _ = plt.figure(figsize=figsize)
        _ = plt.hist(prediction_std, bins=bins)
