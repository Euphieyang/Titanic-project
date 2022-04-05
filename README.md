# Titanic-project objective
1. Find an appropriate model for data prediction<br>
2. Find out the importance of features<br>
<br>
In order to select an fit model for the data, I choses Logistic Regression, Decision Tree Classifier, RandomForest Classifier to test the fitness. <br>
Based on ROC curve, I found that the best performance of the model is random forest.<br>
Therefore, this dataset is recommended to apply random forest to predict the survival rate.<br>
<br>
For the feature importance, I apply KBest and ExtraTreesClassifier, both are from sklearn to help me find out which feature is important for the model.<br>
The results from the two model are different, but both of them select Sex_male and Fare as important facors.<br>
Thus, gender and fare are vital for presengers' survival. If we need to simplify the model to predict, we could not neglect these two factors!<br>
