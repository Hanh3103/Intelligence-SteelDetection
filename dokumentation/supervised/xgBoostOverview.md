# XGBoost (Extreme Gradient Boosting)
Implemented by Jona

XGBoost is an ensemble learning algorithm based on decision trees as weak learners. Each decision tree is created to correct the error of its parent. XGBoost uses gradient descent to optimize the loss function and regularization to prevent overfitting. In other multi-classification problems, XGBoost has proven to be a robust and accurate model, which is why we decided to apply it.

## XGBoost-Classifier:

In the following, we go over parameters that are useful in our case:
- `n_estimators` (default = 100): the number of trees to build
- `learning_rate` (default = 0.1): factor scaling the contribution of each tree. The lower the value, the more robust the model (requires a higher number of trees though)
- `max_depth` (default = 3): maximum depth of each tree in the ensemble
- `gamma`** (default = 0): minimum loss reduction required to make further partition of a leaf node
- `reg_lambda`** (default = 1): controls the amount of regularization applied to the weights of the features in each tree
- `objective`**: specifies the learning task
- In our case of multi-classification, the Classifier needs additionally `num_classes`: Number of classes that the data need to be categorized in

To evaluate an initial performance with standard parameters, we do a 9-fold Cross-Validation. The initial Classifier already has an accuracy of 96.33%, which is already a satisfactory performance. To enhance the performance, we do a hyperparameter optimization with GridSearch to determine the best parameters. For the final evaluation of our model with improved parameters, we do another cross-validation and get an accuracy of 97.25%. Further documentation and evaluation can be found in the XGBoost_SupervisedLearning Jupyter Notebook.
