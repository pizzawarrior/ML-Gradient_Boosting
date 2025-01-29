## Gradient Boosting with Logistic Regression

### Intro
This project is an exploration of applying Gradient Boosting with Logistic Regression to a public data set. Analyses have been crafted both in R and Python for comparison. Several plots are included, such as ROC plots, confusion matrices, and losses calculated at a range of thresholds.

### Frameworks Used:
#### Python:
- xgboost
- numpy
- pandas
- sklearn
- matplotlib

#### R:
- xgboost
- caret
- pROC

### About the Data
'German Credit Data', found here:
- https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Number of columns: 21
- Number of rows: 1000
- Number of predictors: 20
- The response is binary, with the following corresponding values: 1 = good, 2 = bad
- Missing data? No

The data set contains a range of predictors used for determining credit-worthiness of customers, including:
- status of bank account
- employment status
- age
- whether the person owns property or not
- etc.

Data Types:
- Integers
- Categorical (as strings)

### Analysis Approach
The data is loaded and some basic EDA is done to check for missing data, ranges of values, outliers, etc.

All categorical (string) predictors are then encoded as binary values, and the response is converted from (1, 2) to (0, 1), where 0 = bad, and 1 = good.

The data was partitioned into training and testing sets using a random 80/20 split.

An initial xgboost model is built using a high number of rounds to test how the logloss value changes at each iteration.

Predictions are made using the test data set, and evaluation metrics such as RMSE, MSE, and accuracy are calculated. A confusion matrix is built as well to help gauge the results.

<img src="https://github.com/user-attachments/assets/859ad782-2d4e-4c5f-a060-cf2011afb0cb" alt="ROC_curve_highest_acc" width="700">

Using all of the predicted probabilities as range of threshold values for converting the probabilities to 1s or 0s, I then calculated the accuracy, true-negative-rate, and true-positive-rate for each threshold. This allows me to locate the threshold that generates the highest accuracy.

<img src="https://github.com/user-attachments/assets/c6fb3386-a283-4aae-95a1-688c36ef7a2a" alt="Loss_vs_Threshold" width="700">\
In the plot above, we can see that the threshold that produces the highest accuracy is around ~ 8.7.

As a further exercise, I explored the following problem:
- Consider that the cost of misclassifying a bad customer as good in this case is 5x higher than classifying a good customer as bad. Return the threshold value that minimizes this cost.
- After finding the optimal threshold value, I then calculated the accuracy and AUC value to explore potential tradeoffs of prioritizing the lowest false-positive-rate over accuracy and other metrics.

### Conclusion
The highest accuracy (using test data from the same sample) was 79%. This is inline with the supplemental modeling information provided with this data set. It is possible that building a cross-validated xgboost model could improve accuracy, but this is beyond the scope of this project.
