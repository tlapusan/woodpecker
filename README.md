
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

![text](https://github.com/tlapusan/woodpecker/blob/version_0.1/resources/docs/images/classification/titanic_train_describe.png)   


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
