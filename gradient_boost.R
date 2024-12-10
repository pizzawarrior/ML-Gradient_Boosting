# A comparison of gradient boosting with logistic regression on a test data set

library(xgboost)



df <- read.table('~/CODING/ml-gradient-boost/germancredit.txt', header = F)

summary(df)

sum(is.na(df)) # 0

?xgboost
