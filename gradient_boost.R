# A comparison of gradient boosting with logistic regression on a test data set

# Resources:
# Data source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
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
class(xgb_train) # "xgb.DMatrix"

# define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

# fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, 
                  watchlist = watchlist, 
                  nrounds = 20, 
                  objective = "binary:logistic")
# we get our lowest test rmse @ run 15, so let's use that for the final model

final = xgboost(data = xgb_train, nrounds = 13, objective = "binary:logistic")

pred <- predict(final, xgb_test)
head(pred) # need to convert to binary

prediction <- as.numeric(pred > 0.5) # if num > 0 convert to 1
head(prediction)

mse <- mean((test_y - prediction)^2) # MSE 0.24
mae <- mean(abs(test_y - prediction)) # .24
# NOTE the reason why mse = mae is because of the binary classification values.
# When we square each difference the result is always 0 or 1, which is the same as 
# the absolute differences

rmse <- sqrt(mse) # 0.49

?confusionMatrix

conf_matrix <- confusionMatrix(as.factor(prediction), as.factor(test_y), dnn = c("Model", "Actual"))
conf_matrix
#       Actual
# Model    0    1
#       0  25  14
#       1  34 124

accuracy <- sum(test_y == prediction) / length(test_y)
# Accuracy : 0.7563

?roc
roc_curve <- roc(test_y, prediction)
auc(roc_curve)
plot(roc_curve)

# Area under the curve: 0.6611
# This means our model has a 66% chance of correctly classifying positive and negative credit risks
# This suggests potential room for improvement in the model


# ******************************************************************************
# Let's compute these values manually and compare to the results above

# get unique threshold values from the predicted values
thr = sort(unique(c(pred, 0, 1)), decreasing=TRUE) # include 0 and 1 as thresholds

# set up variables for ROC and AUC
n = length(pred)
y = test_y
pos = length(y[y == 1])  # number of positive values
neg = length(y[y == 0])  # number of negative values
auc = 0
last_tpr = 1
last_tnr = 0

# data frame to store results
res_df = data.frame(Thr=double(), TNR=double(), TPR=double(), Acc=double()) # AUC=double(), ltnr=double(), ltpr=double())

# capture TNR, TPR, Accuracy, AUC contribution at each threshold from predicted values
for (num in thr){
  pred_round <- as.integer(pred > num) # if predicted values > thr value convert to 1, else 0
  acc = sum(y == pred_round) / n # accuracy
  tp = sum(y[y == 1] == pred_round[y == 1]) # true pos
  tn = sum(y[y == 0] == pred_round[y == 0]) # true neg
  tpr = tp / pos # sensitivity
  tnr = tn / neg # specificity
  
  # calc AUC contribution
  if (num < 1){
    auc = auc + (last_tpr * (last_tnr - tnr))
  }
  
  df = data.frame(Thr = i, TNR = tnr, TPR = tpr, Acc = acc)
  res_df = rbind(res_df, df)
  last_tpr = tpr
  last_tnr = tnr
}

auc # 0.7710233

# plot ROC
plot(res_df$TNR, res_df$TPR, type = 'l', xlim = c(1.002, 0), ylim = c(0, 1.002), 
     yaxs = "i", xaxs = "i", col = 'blue', ylab = 'Sensitivity (TPR)', xlab = 'Specificity (TNR)', main = 'ROC curve')
abline(1, -1, col = 'gray')
legend('center', legend = paste('AUC = ', round(auc * 100, 2)), bty = 'n')

tail(res_df, 100)

which.max(res_df$Acc) # find threshold that returns highest accuracy 
#     Thr        TNR        TPR        Acc
# 148 0.57248229 0.59322034 0.8840580 0.7969543

# Summary: A threshold value of .57 gives us the highest classification accuracy


# ******************************************************************************

# Alternative to above, using more tunable features
# TODO Now let's do this again and define a params object like the docs
# use cross validation and verbose

rm(list = ls())

options(scipen = 999)
set.seed(123)

df <- read.table('~/CODING/ml-gradient-boost/germancredit.txt', header = F)
