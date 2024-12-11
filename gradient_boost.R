# A comparison of gradient boosting with logistic regression on a test data set
# We could use one-hot encoding, but we'll try factors first

# Resources:
# https://www.geeksforgeeks.org/ml-one-hot-encoding/
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html


library(xgboost)
library(caret)
library(pROC)

rm(list = ls())

options(scipen = 999)
set.seed(123)

df <- read.table('~/CODING/ml-gradient-boost/germancredit.txt', header = F)

summary(df)

sum(is.na(df)) # 0

?xgboost

# We need binary values of 0 or 1 as the response for the model
# Note that in this dataset the response variable value of 1 = good, and 2 = bad.
# Normally it's 0 = bad, and 1 = good. We need to flip these and provide the right values for the model
df$V21[df$V21 == 2] <- 0 # change 2 to 0

# convert num to int
# df$V21 <- as.integer(df$V21)
class(df$V21)

# convert cols to factors 
categoricals <- c('V1', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V15', 'V17', 'V19', 'V20')

df[, categoricals] <- lapply(df[, categoricals], as.factor)
summary(df)

contrasts(df$V1) # Check how col V1 was split out into new binary columns

index <- createDataPartition(df$V1, p = 0.8, list = FALSE)
train_df <- df[index, ]
test_df <- df[-index, ]

train_x = data.matrix(train_df[, -21]) # keep response out of training data
train_y = train_df[,21]

test_x = data.matrix(test_df[, -21]) # keep response out of test data
test_y = test_df[, 21]

# define training and testing sets for xgb.train
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

dim(xgb_train)
class(xgb_train)

# define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

# fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, watchlist = watchlist, nrounds = 70, objective = "binary:logistic")
# we get our lowest test rmse @ run 15, so let's use that for the final model

final = xgboost(data = xgb_train, nrounds = 13, objective = "binary:logistic")

pred <- predict(final, xgb_test)
head(pred) # need to convert to binary

prediction <- as.numeric(pred > 0.5) # if num > 0 convert to 1
head(prediction)

# TODO: Consider that the cost of classifying bad credit risks as good (a score of 1) 
# is higher than classifying good as bad. Write a function to loop thru different 
# rounding thresholds (.6, .65, .7 , .75 ...) that make it harder to classify bad as good
# return a table of MSE, MAE, RMSE values for each threshold value

mean((test_y - prediction)^2) # MSE 0.24
MAE(test_y, prediction) # MAE 0.24
RMSE(test_y, prediction) # RMSE 0.49

conf_matrix <- confusionMatrix(as.factor(prediction), as.factor(test_y))
conf_matrix
#           Reference
# Prediction   0    1
#           0  25  14
#           1  34 124

# Accuracy : 0.7563

roc_curve <- roc(test_y, prediction)
auc(roc_curve)
plot(roc_curve)

# Area under the curve: 0.6611
# This means our model has a 66% chance of correctly classifying positive and negative credit risks
# This suggests potential room for improvement in the model


# TODO Now let's do this again and define a params object like the docs
# use cross validation and verbose
