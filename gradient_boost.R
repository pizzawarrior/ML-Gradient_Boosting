# A comparison of gradient boosting with logistic regression on a test data set
# We could use one-hot encoding, but we'll try factors first

# Resources:
# https://www.geeksforgeeks.org/ml-one-hot-encoding/
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html


library(xgboost)
library(caret)

rm(list = ls())

options(scipen = 999)
set.seed(123)

df <- read.table('~/CODING/ml-gradient-boost/germancredit.txt', header = F)

summary(df)

sum(is.na(df)) # 0

?xgboost

# Approach: 
# Convert all binary and categorical variables to factors
# Convert response to 0, 1 binary
# Separate into train and test data: 75/ 25
# Note: Test df should not have a response column (this may or may not be true)

# Because we need a binary value of 1 or 0 for the model, let's do some quick substitutions 
# to be inline with what the model expects
# Also note that in this dataset the response variable value of 1 = good, and 2 = bad.
# Normally it's 0 = bad, and 1 = good. We need to flip these and provide the right values
df$V21[df$V21 == 2] <- 0 # change 2 to 0

# convert num to int
# df$V21 <- as.integer(df$V21)
class(df$V21)

# convert cols to factors 
categoricals <- c('V1', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V15', 'V17', 'V19', 'V20')

df[, categoricals] <- lapply(df[, categoricals], as.factor)

is.factor(df$V21)
contrasts(df$V1) # Check how col V1 was split out into new binary columns

index <- createDataPartition(df$V1, p = 0.8, list = FALSE)
train_df <- df[index, ]
test_df <- df[-index, ]

str(train_df)

train_x = data.matrix(train_df[, -21])
train_y = train_df[,21]

test_x = data.matrix(test_df[, -21])
test_y = test_df[, 21]

# define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

dim(xgb_train)
class(xgb_train)


# define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

# fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, watchlist = watchlist, nrounds = 70, objective = "binary:logistic")
# we get our lowest test rmse @ run 15, so let's use that for the final model

final = xgboost(data = xgb_train, nrounds = 13)

pred <- predict(final, xgb_test)
pred

# TODO: Don't we need to round the numbers to binary?

mean((train_y - pred)^2) # MSE
MAE(test_y, pred) # MAE
RMSE(test_y, pred) # RMSE


# TODO Now let's do this again and define a params object like the docs

