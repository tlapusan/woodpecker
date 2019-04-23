
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

> features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"] <br>
> target = "Survived" 

Let's see some descriptive statistics about training set. <br> 
> train[features].describe() 

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/titanic_train_describe.png)
   
### Train the model 
> model = DecisionTreeClassifier(criterion="entropy", random_state=random_state, min_samples_split=20)
> model.fit(train[features], train[target])

### Start using the library

> dts = DecisionTreeStructure(model, train, features, target)

#### Visualize feature importance

You don't have to type all the code needed to extract feature importance,
to map them to feature names and to sort them.
Now, you just type this simple utility function. 

> dts.show_features_importance() 

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/feature_importance.png)

#### Visualize decision tree structure 

Like in the above case, this function is also an utility function what 
wrap all the code needed to visualize decision tree structure using graphviz.

> dts.show_decision_tree_structure() 

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/decision_tree_structure.png)

#### Leaves impurity distribution

Impurity is a metric which shows how confident is your leaf prediction. <br>
In case of entropy, impurity is a range of values between 0 and 1. 
0 means that the leaf node is very confident about its predictions, 1 means the opposite.

The tree performance is directly influenced by each leaf performance. So it's very important to have a general 
overview of how leaves impurity look.

> dts.show_leaf_impurity_distribution(bins=40, figsize=(20, 7))

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_impurity_distribution.png)

#### Leaves sample distribution

Sample is a metric which shows how many examples from training set reached that node. <br>
For a leaf is ideal to have an impurity very close to 0, but it's also equally important 
to have a significant set of samples reaching that leaf. If the set of samples is very small, could be a sign 
of outfitting for the leaf.

That's why is important to look both at leaves impurity (previous plot) and samples to get a better understanding of tree performance.

> dts.show_leaf_samples_distribution(bins=40, figsize=(20, 7))

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_sample_distribution.png)

#### Individual leaves metrics

There could be the case when we want to investigate individual leaf behavior. <br>
We could analyze leaves with very good, medium or very low performance.  


> plt.figure(figsize=(40,30)) <br>
> plt.subplot(3,1,1) <br>
> dts.show_leaf_impurity() <br>

> plt.subplot(3,1,2) <br>
> dts.show_leaf_samples()

> plt.subplot(3,1,3)
> dts.show_leaf_samples_by_class()

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/leaves_metrics.png)

#### Get node samples
This function return a dataframe with all training samples reaching a node.
After looking at individual leaves metrics, we can see that there are some interesting leaves. 
For example the leaf 19 has impurity 0, a lot of samples and all people survived (survived=1)
Getting the samples from such a leaf can help us to discover patterns in data or to discover why a leaf 
has good/bad performance.

> dts.get_node_samples(node_id=19)[features + [target]].describe()

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/get_node_samples.png)

We can see that majority of people were from a high social economic status (Pclass = 1), most of them were young to mid age,
bought an expensive ticket (mean(Fare) from training is 32) and are all women.

#### Visualize decision tree path prediction
There will be moments when we need to justify why our model predicted a specific value.
Looking at the whole tree and tracking the path prediction is not time effective if the depth of the tree is large.

Let's look at prediction path for the following sample : 
>Pclass             3.0 <br>
Age               28.0 <br>
Fare              15.5 <br>
Sex_label          0.0 <br>
Cabin_label       -1.0 <br>
Embarked_label     1.0 <br>

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/decision_tree_prediction_path.png)

#### Visualize decision tree splits path prediction
This visualization shows the training data splits the model was build. 
It can be used also as a way to learn how decision tree was built.

The sample is the same as above. 
> dts.show_decision_tree_splits_prediction(sample, bins=20)

![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/decision_tree_splits_prediction_part_1.png)
![](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/decision_tree_splits_prediction_part_2.png)




# Release History
- 0.1
    -  model structure investigation for DecisionTreeClassifier 

# Meta
Tudor Lapusan <br>
twitter : @tlapusan <br> 
email : tudor.lapusan@gmail.com

# Library dependencies

- jupyter
- matplotlib 
- scikit-learn 
- pandas 

# License
This project is licensed under the terms of the MIT license, see LICENSE.

