
# Purpose 
A python library used for model structure interpretation. <br>
Right now the library contains logic for [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) from [scikit-learn](https://scikit-learn.org/stable/). 
Next versions of the library will contain other types of algorithms, like [DecisionTreeRegression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor), 
[RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests).


Becoming a better machine learning engineer is important to understand more deeply the model structure and also to have an intuition of what is happening if we change the model inputs, how these will reflect in model performance. 
By model inputs we mean to add more data, add new features and to change model hyperparameters


This library was developed with two main ideas in mind :
- help us better understand the model structure, the model results and based on this to properly choose others hyperparameter values, other set of features for the next iteration
- to justify/explain the predictions of ML models both for technical and non technical people

# Usage example

### Training example
The well known [titanic dataset](https://www.kaggle.com/c/titanic/data) was chosen to show library capabilities.

features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"] <br>
target = "Survived" 

Let see some descriptive statistics about training set. <br> 
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/titanic_train_describe.png)
   
### Train the model 
model = DecisionTreeClassifier(criterion="entropy", random_state=random_state, min_samples_split=20)
model.fit(train[features], train[target])

### Start using the library

dts = DecisionTreeStructure(model, train, features, target)

#### Visualize feature importance

You don't have type all the code needed to extract feature importance,
to map them to feature names and to sort them.
Now, you just type this simple utility function. 

dts.show_features_importance()
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/feature_importance.png)

#### Visualize decision tree structure 

Like in the above case, this function is also an utility function what 
wrap all the code needed to visualize decision tree structure using graphviz.

dts.show_decision_tree_structure()
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/decision_tree_structure.png)

#### Leaves impurity distribution

Impurity is a metric which shows how confident is your leaf prediction. <br>
In case of entropy, impurity is a range of values between 0 and 1. 
0 means that the leaf node is very confident about its predictions, 1 means the opposite.

The tree performance is directly influenced by each leaf performance. So it's very important to have a general 
overview of how leaves impurity looks.

dts.show_leaf_impurity_distribution(bins=40, figsize=(20, 7))
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_impurity_distribution.png)

#### Leaves sample distribution

Sample is a metrics which shows how many examples from training set reached that node. <br>
For a leaf is ideal to have an impurity very close to 0, but it's also equally important 
to have a significant set of samples reaching that leaf. If the set of samples is very small, could be a sign 
of outfitting for the leaf.

That's why is important to look both at leaves impurity and samples to get a better understanding of tree performance.

dts.show_leaf_samples_distribution(bins=40, figsize=(20, 7))
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_sample_distribution.png)

#### Individual leaves metrics

There could be the case when we want to investigate individual leaf behavior. <br>
We could analyze leaves with very good, medium or very low performance.  


plt.figure(figsize=(40,30))
plt.subplot(3,1,1)
dts.show_leaf_impurity()

plt.subplot(3,1,2)
dts.show_leaf_samples()

plt.subplot(3,1,3)
dts.show_leaf_samples_by_class()

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_metrics.png)




# Release History
- 0.1
    -  model structure investigation for DecisionTreeClassifier 

# Meta
Tudor Lapusan <br>
twitter : @tlapusan <br> 
email : tudor.lapusan@gmail.com

# License
This project is licensed under the terms of the MIT license, see LICENSE.

 


TODO
- visualize decision path
- visualize node/leaf impurity (as matplolib)
- visualize node/leaf samples
- visualize decision path, leaf, all nodes splits (ex. histograms - like in animl)
