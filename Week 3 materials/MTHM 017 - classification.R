# ---
# title: |
# Advanced Topics in Statistics  
# Classification in R and Python

# date: "03/02/2021"
# output: pdf_document
# ---
  
# ```{r setup, include=FALSE}
# knitr::opts_chunk$set(echo = TRUE)
# ```

# In this practical we will cover the following topics:
  
# * Logistic regression



## Classification using logistic regression

# **Quick recap** - for more details see Classification slides on the ELE page.

# Imagine that an observation consists of some explanatory variables $X_1,X_2,\dots,X_p$ and a binary 
# label $Y$ (that is $Y$ is either 1 or 0). When doing classification, we want to find a method that 
# assigns one label to each observation in a way that the method correctly predicts not only the observed 
# labels, but it also performs well when assingning labels to new observations.

# Logistic regression is one method that can used to classify observations. It uses the form
# \[p:=\mathbb{P}(Y=1|X_1,\dots,X_p)=\frac{\mathrm{e}^{\beta_0+\beta_1X_1+\dots+\beta_pX_p}}{1+\mathrm{e}^{\beta_0+\beta_1X_1+\dots+\beta_pX_p}},\]
# that is we first estimate the probability that the observation (with explanatory variables $X_1,\dots,X_p$) is of category 1.
# The coefficients $\beta_0,\beta_1,\dots,\beta_p$ are estimated using the observed data.

# Then, in order to predict the label of new observations, we choose a threshold $p*$, and assign label 1 
# if the value $p$ of the new observation is above the threshold $p*$ (and assing label 0 otherwise).

# There are a couple of measures of testing a model’s classification performance. Using our data we can 
# construct the following *confusion matrix*:

# \begin{center}
# \begin{tabular}{c | c c}
# & Obs:0 & Obs:1\\\hline
# Pred:0 & a & b\\
# Pred:1 & c & d
# \end{tabular}
# \end{center}

# The *accuracy* is the fraction of correct classifications, that is $(a+d)/(a+b+c+d)$. 
# Furthermore we can define the following measures:
  
# * Sensitivity: proportion of true positives predicted as such, $a/(b+d)$.

# * Specificity: proportion of true negatives predicted as such, $a/(a+c)$.

# * Positive predictive value: proportion of predicted positives that are actually correct, $d/(c+d)$.

# * Negative predictive value: proportion of predicted negatives that are actually correct, $a/(a+b)$.


# **Breast Cancer data**
  
# In this example we will attempt to classify tumors as being benign (0) or malignant (1). The data are 
# readily available in Python, in R you will have to read in the `cancer.csv` dataset, which you can find 
# on the ELE page. The data consists of 30 explanatory variables (different measurements of the tumor), 
# and the label indicating whether the tumor is benign or malignant.

# Since we only have one dataset, to test model performance we will randomly split to data, and build 
# the model on 75\% of available, and test the model on the remaining 25\%.

# \textcolor{red}{\textbf{R:}}

# Load the necessary packages and the data:
# ```{r,eval=FALSE}
# The caret and e1071 package is required for using the confusionMatrix call
library(dplyr)
library(caret)
library(e1071)
library(MASS)
library(tidyverse)
library(readr)
library(repr)
library(glmnet)

cancer <- read.csv("cancer.csv",header=FALSE)
# ```

# Look at the data (there is a lot so just the first three variables here, try altering this to get a sense of the data)
# ```{r}
y <- cancer[,31]
v1 <- cancer[,1]
v2 <- cancer[,2]
v3 <- cancer[,3]
v4 <- cancer[,4]
v5 <- cancer[,5]
v6 <- cancer[,6]
# Basic Scatterplot Matrix
pairs(~y+v1+v2+v3,data=cancer,main="Simple Scatterplot Matrix")
pairs(~y+v4+v5+v6,data=cancer,main="Simple Scatterplot Matrix")
# ```


# Next we manually define a function that will split the data into training set and test set, and use 
# this function to make the split:
# ```{r,eval=FALSE}
# Define function that will split the data into training and test sets
trainTestSplit <- function(df,trainPercent,seed1){
  ## Sample size percent
  smp_size <- floor(trainPercent/100 * nrow(df))
  ## set the seed 
  set.seed(seed1)
  train_ind <- sample(seq_len(nrow(df)), size = smp_size)
  train_ind
}

# Split as training and test sets
train_ind <- trainTestSplit(cancer,trainPercent=85,seed=123)
train <- cancer[train_ind, ]
test <- cancer[-train_ind, ]

# ```

# Then we fit the glm (note that the labels are in the last column of the dataset - V31).

# Explore differing numbers of the tumor measurements to aid with classification
# ```{r,eval=FALSE}
# Fit a generalized linear logistic model
# Note that the last column (V31) contains the labels

# This fits all the 30 measurements
fit_all <- glm(V31~.,family=binomial,data=train,control = list(maxit = 50))
summary(fit_all)
# Coefficients:
#             Estimate Std. Error   z value   Pr(>|z|)
# (Intercept)  8.677e+03  2.389e+07   0.000    1.000
# V1           5.868e+03  1.316e+07   0.000    1.000
# V2          -1.031e+02  3.379e+05   0.000    1.000
# V3          -4.465e+02  2.337e+06   0.000    1.000
# V4          -2.632e+01  3.965e+04  -0.001    0.999
# V5          -3.916e+04  1.760e+08   0.000    1.000
# V6           4.714e+04  5.696e+07   0.001    0.999
# V7          -4.167e+03  4.085e+07   0.000    1.000
# V8          -3.360e+04  7.480e+07   0.000    1.000
# V9           1.099e+04  2.468e+07   0.000    1.000
# V10         -7.143e+04  9.127e+07  -0.001    0.999
# V11          1.533e+03  1.838e+07   0.000    1.000
# V12         -3.374e+01  1.668e+06   0.000    1.000
# V13          4.217e+02  2.997e+06   0.000    1.000
# V14         -6.404e+01  1.407e+05   0.000    1.000
# V15          7.213e+04  1.613e+08   0.000    1.000
# V16         -6.576e+04  1.095e+08  -0.001    1.000
# V17          4.245e+04  6.974e+07   0.001    1.000
# V18         -2.003e+05  3.771e+08  -0.001    1.000
# V19          4.725e+04  1.497e+08   0.000    1.000
# V20          4.738e+05  6.476e+08   0.001    0.999
# V21         -2.664e+03  4.779e+06  -0.001    1.000
# V22         -5.037e+00  2.008e+05   0.000    1.000
# V23          4.284e+01  1.895e+05   0.000    1.000
# V24          1.728e+01  3.218e+04   0.001    1.000
# V25          3.789e+03  6.143e+07   0.000    1.000
# V26          2.994e+03  2.434e+07   0.000    1.000
# V27         -7.080e+03  2.522e+07   0.000    1.000
# V28          5.651e+03  4.429e+07   0.000    1.000
# V29         -1.056e+04  1.650e+07  -0.001    0.999
# V30         -1.330e+04  5.061e+07   0.000    1.000

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 6.2537e+02  on 482  degrees of freedom
# Residual deviance: 2.3388e-09  on 452  degrees of freedom
# AIC: 62


# This fits V1
fit_V1 <- glm(V31~V1,family=binomial,data=train,control = list(maxit = 50))
summary(fit_V1)
# Coefficients:
#             Estimate Std. Error   z value Pr(>|z|)    
# (Intercept)  14.6188     1.3715   10.66   <2e-16 ***
# V1           -0.9831     0.0963  -10.21   <2e-16 ***

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.37  on 482  degrees of freedom
# Residual deviance: 281.18  on 481  degrees of freedom
# AIC: 285.18


# This fits V2
fit_V2 <- glm(V31~V2,family=binomial,data=train,control = list(maxit = 50))
summary(fit_V2)
# Coefficients:
#             Estimate Std. Error   z value Pr(>|z|)    
# (Intercept)  5.24110    0.57566   9.105   <2e-16 ***
# V2          -0.23377    0.02829  -8.263   <2e-16 ***

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.37  on 482  degrees of freedom
# Residual deviance: 536.93  on 481  degrees of freedom
# AIC: 540.93

# This fits V3
fit_V3 <- glm(V31~V3,family=binomial,data=train,control = list(maxit = 50))
summary(fit_V3)
# Coefficients:
#             Estimate Std. Error   z value Pr(>|z|)    
# (Intercept) 15.03237    1.42038   10.58   <2e-16 ***
# V3          -0.15570    0.01533  -10.15   <2e-16 ***

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.37  on 482  degrees of freedom
# Residual deviance: 261.21  on 481  degrees of freedom
# AIC: 265.21


# This fits V1 + V2 + V3
fit_V1_V2_V3 <- glm(V31~V1+V2+V3,family=binomial,data=train,control = list(maxit = 50))
summary(fit_V1_V2_V3)
# Coefficients:
#           Estimate Std. Error z   value Pr(>|z|)    
# (Intercept) 17.15217    1.97095   8.702  < 2e-16 ***
# V1           5.62102    1.01587   5.533 3.14e-08 ***
# V2          -0.22255    0.04759  -4.677 2.92e-06 ***
# V3          -0.99542    0.15733  -6.327 2.50e-10 ***

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.37  on 482  degrees of freedom
# Residual deviance: 194.11  on 479  degrees of freedom
# AIC: 202.11


# Use backstep to remove all variables with no correlation to V31
fit_backstep <- fit_all %>% stepAIC(trace=FALSE, direction = c("backward"))
coef(fit_backstep)
print(fit_backstep)

# This fits backstep model
fit_backstep1 <- glm(formula = V31 ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
                       V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20 + V21 + V29 + 
                       V30, family = binomial, data = train, control = list(maxit = 50))
summary(fit_backstep1)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)
# (Intercept)  1.023e+04  4.542e+06   0.002    0.998
# V1           6.276e+03  1.149e+06   0.005    0.996
# V2          -3.491e+02  6.164e+04  -0.006    0.995
# V4          -5.405e+01  1.017e+04  -0.005    0.996
# V5          -8.852e+04  2.116e+07  -0.004    0.997
# V6           1.261e+05  2.271e+07   0.006    0.996
# V7          -8.776e+04  1.747e+07  -0.005    0.996
# V8          -2.547e+04  2.661e+07  -0.001    0.999
# V9           4.329e+04  9.110e+06   0.005    0.996
# V10         -1.599e+05  3.400e+07  -0.005    0.996
# V13          8.165e+02  1.803e+05   0.005    0.996
# V14         -1.421e+02  2.667e+04  -0.005    0.996
# V15          2.288e+05  4.840e+07   0.005    0.996
# V16         -1.612e+05  2.955e+07  -0.005    0.996
# V17          1.290e+05  2.433e+07   0.005    0.996
# V18         -6.829e+05  1.293e+08  -0.005    0.996
# V19          1.998e+05  3.602e+07   0.006    0.996
# V20          1.669e+06  3.056e+08   0.005    0.996
# V21         -1.948e+03  3.467e+05  -0.006    0.996
# V29         -4.119e+04  7.478e+06  -0.006    0.996
# V30         -1.196e+05  2.202e+07  -0.005    0.996

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 6.2537e+02  on 482  degrees of freedom
# Residual deviance: 1.0781e-08  on 462  degrees of freedom
# AIC: 42
# ```

train
# Dumy code categorical predictor variables
x <- model.matrix(V31~., train)[,-1]
# Convert the outcome (class) to a numerical variable
y_V31 <- train$V31

# Use lasso regression to remove all variables with no correlation to V31
cv.lasso <- cv.glmnet(x = x, y_V31, alpha = 1, family = "binomial")
# Fit the final model on the training data
lasso.model <- glmnet(x = x, y_V31, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
# Display regression coefficients
coef(lasso.model)
#               s0
# (Intercept)  29.1171699
# V1            .        
# V2            .        
# V3            .        
# V4            .        
# V5            .        
# V6            .        
# V7            .        
# V8          -33.9464254
# V9            .        
# V10          31.6135072
# V11          -9.0464604
# V12           0.5784591
# V13           .        
# V14           .        
# V15         -92.9503483
# V16          42.6760739
# V17           .        
# V18           .        
# V19           .        
# V20          77.0319339
# V21          -0.7109714
# V22          -0.2669398
# V23           .        
# V24           .        
# V25         -18.2286258
# V26           .        
# V27          -5.3631108
# V28         -18.7103130
# V29          -8.3093734
# V30 

# Make predictions on the test data
lasso.test <- model.matrix(V31 ~., data = test)[,-1]
lasso.probs <- lasso.model %>% predict(newx = lasso.test)
predicted.classes <- ifelse(lasso.probs > 0.5, 1, 0)
print(lasso.model)

# This fits lasso1 model
fit_lasso1 <- glm(formula = V31 ~ V8 + V10 + V11 + V12 + V15 + V16 + V20 + V21 + V22 + V25 + V27 + V28 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso1)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   50.2028    16.0646   3.125  0.00178 ** 
# V8           -52.4697    45.5220  -1.153  0.24907    
# V10           19.9787   156.2735   0.128  0.89827    
# V11          -18.8776     6.4815  -2.913  0.00359 ** 
# V12            2.1273     1.4677   1.449  0.14724    
# V15         -270.2908   198.6908  -1.360  0.17372    
# V16          100.3277    61.2341   1.638  0.10133    
# V20          291.8072   650.6440   0.448  0.65380    
# V21           -1.1237     0.4299  -2.614  0.00894 ** 
# V22           -0.5026     0.1411  -3.561  0.00037 ***
# V25          -31.1629    37.4893  -0.831  0.40583    
# V27          -11.6342     4.5188  -2.575  0.01004 *  
# V28          -26.6205    26.6951  -0.997  0.31866    
# V29          -14.2849     9.1415  -1.563  0.11813    

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.37  on 482  degrees of freedom
# Residual deviance:  48.14  on 469  degrees of freedom
# AIC: 76.14


# This fits lasso2 model, removing V10
fit_lasso2 <- glm(formula = V31 ~ V8 + V11 + V12 + V15 + V16 + V20 + V21 + V22 + V25 + V27 + V28 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso2)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   51.6029    11.9605   4.314  1.6e-05 ***
# V8           -50.8782    43.7221  -1.164 0.244557    
# V11          -18.9225     6.4713  -2.924 0.003455 ** 
# V12            2.1326     1.4678   1.453 0.146265    
# V15         -283.3803   171.6716  -1.651 0.098798 .  
# V16           99.4678    61.0974   1.628 0.103521    
# V20          333.5617   569.9192   0.585 0.558360    
# V21           -1.1524     0.3711  -3.105 0.001903 ** 
# V22           -0.5048     0.1404  -3.597 0.000322 ***
# V25          -28.9058    33.1383  -0.872 0.383057    
# V27          -11.6141     4.5093  -2.576 0.010008 *  
# V28          -27.2447    26.1834  -1.041 0.298092    
# V29          -14.2254     9.1795  -1.550 0.121217    

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.372  on 482  degrees of freedom
# Residual deviance:  48.156  on 470  degrees of freedom
# AIC: 74.156


# This fits lasso3 model, removing V20
fit_lasso3 <- glm(formula = V31 ~ V8 + V11 + V12 + V15 + V16 + V21 + V22 + V25 + V27 + V28 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso3)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   53.1470    11.8230   4.495 6.95e-06 ***
# V8           -52.4617    42.9320  -1.222 0.221718    
# V11          -18.7627     6.2396  -3.007 0.002638 ** 
# V12            2.0260     1.4271   1.420 0.155718    
# V15         -232.7488   142.2723  -1.636 0.101852    
# V16          128.1262    40.5415   3.160 0.001576 ** 
# V21           -1.2240     0.3566  -3.432 0.000598 ***
# V22           -0.4963     0.1348  -3.681 0.000233 ***
# V25          -32.1342    32.5951  -0.986 0.324202    
# V27          -11.7263     4.5477  -2.578 0.009924 ** 
# V28          -25.0542    26.3719  -0.950 0.342096    
# V29          -15.0637     9.4306  -1.597 0.110196    

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.372  on 482  degrees of freedom
# Residual deviance:  48.612  on 471  degrees of freedom
# AIC: 72.612


# This fits lasso4 model, removing V28
fit_lasso4 <- glm(formula = V31 ~ V8 + V11 + V12 + V15 + V16 + V21 + V22 + V25 + V27 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso4)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   52.8511    11.5617   4.571 4.85e-06 ***
# V8           -78.6160    33.8530  -2.322 0.020218 *  
# V11          -17.8353     5.9409  -3.002 0.002681 ** 
# V12            2.2482     1.3755   1.634 0.102159    
# V15         -220.5915   138.9737  -1.587 0.112447    
# V16          130.5217    40.3474   3.235 0.001217 ** 
# V21           -1.2391     0.3476  -3.565 0.000364 ***
# V22           -0.4974     0.1314  -3.785 0.000153 ***
# V25          -41.0096    31.1644  -1.316 0.188204    
# V27          -13.2967     4.2268  -3.146 0.001656 ** 
# V29          -16.2513     8.9861  -1.808 0.070531 .  

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.372  on 482  degrees of freedom
# Residual deviance:  49.579  on 472  degrees of freedom
# AIC: 71.579


# This fits lasso5 model, removing V25, with lowest AIC, with most p-values < 0.05
fit_lasso5 <- glm(formula = V31 ~ V8 + V11 + V12 + V15 + V16 + V21 + V22 + V27 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso5)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   45.3694     8.8215   5.143 2.70e-07 ***
# V8           -94.7792    32.9201  -2.879 0.003988 ** 
# V11          -15.6728     4.8655  -3.221 0.001276 ** 
# V12            2.1651     1.3243   1.635 0.102065    
# V15         -297.4224   140.8381  -2.112 0.034703 *  
# V16          139.9884    38.9027   3.598 0.000320 ***
# V21           -1.0503     0.2978  -3.526 0.000421 ***
# V22           -0.4719     0.1172  -4.026 5.68e-05 ***
# V27          -13.3711     3.7592  -3.557 0.000375 ***
# V29          -20.2970     9.2957  -2.183 0.029000 *  

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.372  on 482  degrees of freedom
# Residual deviance:  51.403  on 473  degrees of freedom
# AIC: 71.403


# This fits lasso6 model, removing V12 - all variables have p-values < 0.05, but AIC is higher than lasso5 model
fit_lasso6 <- glm(formula = V31 ~ V8 + V11 + V15 + V16 + V21 + V22 + V27 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
summary(fit_lasso6)
# Coefficients:
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   47.0734     8.9004   5.289 1.23e-07 ***
# V8           -90.7944    30.6465  -2.963 0.003050 ** 
# V11          -12.1970     3.4151  -3.571 0.000355 ***
# V15         -306.1695   125.7493  -2.435 0.014902 *  
# V16          155.9177    39.0530   3.992 6.54e-05 ***
# V21           -1.2222     0.2713  -4.505 6.62e-06 ***
# V22           -0.3500     0.0779  -4.493 7.04e-06 ***
# V27          -14.1530     3.7577  -3.766 0.000166 ***
# V29          -24.1029     9.4290  -2.556 0.010580 *  

# (Dispersion parameter for binomial family taken to be 1)

# Null deviance: 625.372  on 482  degrees of freedom
# Residual deviance:  54.242  on 474  degrees of freedom
# AIC: 72.242


# Explore the model output

# ```{r}
fitted.model1 <- fitted(fit_all)

# Plot fit_all against V1, V2, V3 and V31
plot(train[,1],fitted(fit_all))
plot(train[,2],fitted(fit_all))
plot(train[,3],fitted(fit_all))
plot(train[,31],fitted(fit_all))

# Plot fit_V1 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_V1))
plot(train[,2],fitted(fit_V1))
plot(train[,3],fitted(fit_V1))
plot(train[,31],fitted(fit_V1))

# Plot fit_V2 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_V2))
plot(train[,2],fitted(fit_V2))
plot(train[,3],fitted(fit_V2))
plot(train[,31],fitted(fit_V2))

# Plot fit_V3 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_V3))
plot(train[,2],fitted(fit_V3))
plot(train[,3],fitted(fit_V3))
plot(train[,31],fitted(fit_V3))

# Plot fit_V1_V2_V3 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_V1_V2_V3))
plot(train[,2],fitted(fit_V1_V2_V3))
plot(train[,3],fitted(fit_V1_V2_V3))
plot(train[,31],fitted(fit_V1_V2_V3))

# Plot fit_backstep1 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_backstep1))
plot(train[,2],fitted(fit_backstep1))
plot(train[,3],fitted(fit_backstep1))
plot(train[,31],fitted(fit_backstep1))

# Plot fit_lasso1 against V1, V2, V3 and V31
plot(train[,1],fitted(fit_lasso1))
plot(train[,2],fitted(fit_lasso1))
plot(train[,3],fitted(fit_lasso1))
plot(train[,31],fitted(fit_lasso1))

# Plot fit_lasso5 against V8, V11, V12, V15, V16, V21, V22, V27, V29 and V31
plot(train[,8],fitted(fit_lasso5))
plot(train[,11],fitted(fit_lasso5))
plot(train[,12],fitted(fit_lasso5))
plot(train[,15],fitted(fit_lasso5))
plot(train[,16],fitted(fit_lasso5))
plot(train[,21],fitted(fit_lasso5))
plot(train[,22],fitted(fit_lasso5))
plot(train[,27],fitted(fit_lasso5))
plot(train[,29],fitted(fit_lasso5))
plot(train[,31],fitted(fit_lasso5))

# Plot fit_lasso6 against V8, V11, V15, V16, V21, V22, V27, V29 and V31
plot(train[,8],fitted(fit_lasso6))
plot(train[,11],fitted(fit_lasso6))
plot(train[,15],fitted(fit_lasso6))
plot(train[,16],fitted(fit_lasso6))
plot(train[,21],fitted(fit_lasso6))
plot(train[,22],fitted(fit_lasso6))
plot(train[,27],fitted(fit_lasso6))
plot(train[,29],fitted(fit_lasso6))
plot(train[,31],fitted(fit_lasso6))
# ```



# Next we check the performance of the trained model (fit_all). Here we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}
# Predict the output from the model
a_all <- predict(fit_all,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_all <- ifelse(a_all>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_all),as.factor(train$V31))
# Accuracy : 1          
# 95% CI : (0.9924, 1)
# No Information Rate : 0.6501     
# P-Value [Acc > NIR] : < 2.2e-16  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         
                                     
#            Sensitivity : 1.0000     
#            Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.3499     
#          Detection Rate : 0.3499     
#    Detection Prevalence : 0.3499     
#       Balanced Accuracy : 1.0000
# ```



# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_all <- predict(fit_all,newdata=test,type="response")
n_all <- ifelse(m_all>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_all),as.factor(test$V31))
# Accuracy : 0.9651          
# 95% CI : (0.9014, 0.9927)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9302          

# Mcnemar's Test P-Value : 0.2482          
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9302          
#          Pos Pred Value : 0.9348          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5349          
#       Balanced Accuracy : 0.9651
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.6). Again, we use the threshold $p*=0.6$.
# ```{r,eval=FALSE}

m_all <- predict(fit_all,newdata=test,type="response")
n_all <- ifelse(m_all>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_all),as.factor(test$V31))
# Accuracy : 0.9651          
# 95% CI : (0.9014, 0.9927)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9302          

# Mcnemar's Test P-Value : 0.2482          
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9302          
#          Pos Pred Value : 0.9348          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5349          
#       Balanced Accuracy : 0.9651
# ```


# Next we check the performance of the trained model (fit_V1). Here we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}
# Predict the output from the model
a_V1 <- predict(fit_V1,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_V1 <- ifelse(a_V1>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_V1),as.factor(train$V31))
#           Reference
# Prediction   0   1
#          0 128  18
#          1  41 296

# Accuracy : 0.8778          
# 95% CI : (0.8453, 0.9057)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.7228          

# Mcnemar's Test P-Value : 0.004181        
                                          
#             Sensitivity : 0.7574          
#             Specificity : 0.9427          
#          Pos Pred Value : 0.8767          
#          Neg Pred Value : 0.8783          
#              Prevalence : 0.3499          
#          Detection Rate : 0.2650          
#    Detection Prevalence : 0.3023          
#       Balanced Accuracy : 0.8500 
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V1 <- predict(fit_V1,newdata=test,type="response")
n_V1 <- ifelse(m_V1>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V1),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 36  1
#          1  7 42

# Accuracy : 0.907          
# 95% CI : (0.8249, 0.959)
# No Information Rate : 0.5            
# P-Value [Acc > NIR] : 7.618e-16      

# Kappa : 0.814          

# Mcnemar's Test P-Value : 0.0771         
                                         
#             Sensitivity : 0.8372         
#             Specificity : 0.9767         
#          Pos Pred Value : 0.9730         
#          Neg Pred Value : 0.8571         
#              Prevalence : 0.5000         
#          Detection Rate : 0.4186         
#    Detection Prevalence : 0.4302         
#       Balanced Accuracy : 0.9070
# ```

# Finally we use the trained model to make predictions for the test dataset. Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V1 <- predict(fit_V1,newdata=test,type="response")
n_V1 <- ifelse(m_V1>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V1),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 37  4
#          1  6 39

# Accuracy : 0.8837          
# 95% CI : (0.7965, 0.9428)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : 5.247e-14       

# Kappa : 0.7674          

# Mcnemar's Test P-Value : 0.7518          
                                          
#             Sensitivity : 0.8605          
#             Specificity : 0.9070          
#          Pos Pred Value : 0.9024          
#          Neg Pred Value : 0.8667          
#              Prevalence : 0.5000          
#          Detection Rate : 0.4302          
#    Detection Prevalence : 0.4767          
#       Balanced Accuracy : 0.8837 
# ```



# Next we check the performance of the trained model (fit_V2). Here we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}
# Predict the output from the model
a_V2 <- predict(fit_V2,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_V2 <- ifelse(a_V2>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_V2),as.factor(train$V31))
#           Reference
# Prediction   0   1
# 0  61  38
# 1 108 276

# Accuracy : 0.6977          
# 95% CI : (0.6546, 0.7384)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : 0.01516         

# Kappa : 0.2653          

# Mcnemar's Test P-Value : 1.127e-08       
                                          
#             Sensitivity : 0.3609          
#             Specificity : 0.8790          
#          Pos Pred Value : 0.6162          
#          Neg Pred Value : 0.7188          
#              Prevalence : 0.3499          
#          Detection Rate : 0.1263          
#    Detection Prevalence : 0.2050          
#       Balanced Accuracy : 0.6200
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V2 <- predict(fit_V2,newdata=test,type="response")
n_V2 <- ifelse(m_V2>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V2),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 13  3
#          1 30 40

# Accuracy : 0.6163          
# 95% CI : (0.5051, 0.7192)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : 0.01993         

# Kappa : 0.2326          

# Mcnemar's Test P-Value : 6.011e-06       
                                          
#             Sensitivity : 0.3023          
#             Specificity : 0.9302          
#          Pos Pred Value : 0.8125          
#          Neg Pred Value : 0.5714          
#              Prevalence : 0.5000          
#          Detection Rate : 0.1512          
#    Detection Prevalence : 0.1860          
#       Balanced Accuracy : 0.6163
# ```

# Finally we use the trained model to make predictions for the test dataset. Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V2 <- predict(fit_V2,newdata=test,type="response")
n_V2 <- ifelse(m_V2>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V2),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 22  5
#          1 21 38

Accuracy : 0.6977          
# 95% CI : (0.5892, 0.7921)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : 0.0001585       

# Kappa : 0.3953          

# Mcnemar's Test P-Value : 0.0032637       
                                          
#             Sensitivity : 0.5116          
#             Specificity : 0.8837          
#          Pos Pred Value : 0.8148          
#          Neg Pred Value : 0.6441          
#              Prevalence : 0.5000          
#          Detection Rate : 0.2558          
#    Detection Prevalence : 0.3140          
#       Balanced Accuracy : 0.6977
# ```



# Next we check the performance of the trained model (fit_V2). Here we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}
# Predict the output from the model
a_V3 <- predict(fit_V3,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_V3 <- ifelse(a_V3>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_V3),as.factor(train$V31))
#           Reference
# Prediction   0   1
#          0  61  38
#          1 108 276

# Accuracy : 0.6977          
# 95% CI : (0.6546, 0.7384)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : 0.01516         

# Kappa : 0.2653          

# Mcnemar's Test P-Value : 1.127e-08       
                                          
#             Sensitivity : 0.3609          
#             Specificity : 0.8790          
#          Pos Pred Value : 0.6162          
#          Neg Pred Value : 0.7188          
#              Prevalence : 0.3499          
#          Detection Rate : 0.1263          
#    Detection Prevalence : 0.2050          
#       Balanced Accuracy : 0.6200
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V3 <- predict(fit_V3,newdata=test,type="response")
n_V3 <- ifelse(m_V3>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V3),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 13  3
#          1 30 40

# Accuracy : 0.6163          
# 95% CI : (0.5051, 0.7192)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : 0.01993         

# Kappa : 0.2326          

# Mcnemar's Test P-Value : 6.011e-06       
                                          
#             Sensitivity : 0.3023          
#             Specificity : 0.9302          
#          Pos Pred Value : 0.8125          
#          Neg Pred Value : 0.5714          
#              Prevalence : 0.5000          
#          Detection Rate : 0.1512          
#    Detection Prevalence : 0.1860          
#       Balanced Accuracy : 0.6163
# ```

# Finally we use the trained model to make predictions for the test dataset. Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V3 <- predict(fit_V3,newdata=test,type="response")
n_V3 <- ifelse(m_V3>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V3),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 22  5
#          1 21 38

# Accuracy : 0.6977          
# 95% CI : (0.5892, 0.7921)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : 0.0001585       

# Kappa : 0.3953          

# Mcnemar's Test P-Value : 0.0032637       
                                          
#             Sensitivity : 0.5116          
#             Specificity : 0.8837          
#          Pos Pred Value : 0.8148          
#          Neg Pred Value : 0.6441          
#              Prevalence : 0.5000          
#          Detection Rate : 0.2558          
#    Detection Prevalence : 0.3140          
#       Balanced Accuracy : 0.6977
# ```



# Next we check the performance of the trained model (fit_V1_V2_V3). Here we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}
# Predict the output from the model
a_V1_V2_V3 <- predict(fit_V1_V2_V3,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_V1_V2_V3 <- ifelse(a_V1_V2_V3>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_V1_V2_V3),as.factor(train$V31))
#           Reference
# Prediction  0  1
#          0 13  3
#          1 30 40

Accuracy : 0.9193         
# 95% CI : (0.8913, 0.942)
# No Information Rate : 0.6501         
# P-Value [Acc > NIR] : <2e-16         

# Kappa : 0.8213         

# Mcnemar's Test P-Value : 0.5218         
                                         
#             Sensitivity : 0.8698         
#             Specificity : 0.9459         
#          Pos Pred Value : 0.8963         
#          Neg Pred Value : 0.9310         
#              Prevalence : 0.3499         
#          Detection Rate : 0.3043         
#    Detection Prevalence : 0.3395         
#       Balanced Accuracy : 0.9078
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V1_V2_V3 <- predict(fit_V1_V2_V3,newdata=test,type="response")
n_V1_V2_V3 <- ifelse(m_V1_V2_V3>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V1_V2_V3),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 35  0
#          1  8 43

Accuracy : 0.907          
# 95% CI : (0.8249, 0.959)
# No Information Rate : 0.5            
# P-Value [Acc > NIR] : 7.618e-16      

# Kappa : 0.814          

# Mcnemar's Test P-Value : 0.01333        
                                         
#             Sensitivity : 0.8140         
#             Specificity : 1.0000         
#          Pos Pred Value : 1.0000         
#          Neg Pred Value : 0.8431         
#              Prevalence : 0.5000         
#          Detection Rate : 0.4070         
#    Detection Prevalence : 0.4070         
#       Balanced Accuracy : 0.9070
# ```

# Finally we use the trained model to make predictions for the test dataset. Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_V1_V2_V3 <- predict(fit_V1_V2_V3,newdata=test,type="response")
n_V1_V2_V3 <- ifelse(m_V1_V2_V3>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_V1_V2_V3),as.factor(test$V31))
#           Reference
# Prediction   0   1
#          0 169   0
#          1   0 314

# Accuracy : 0.9419          
# 95% CI : (0.8695, 0.9809)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : < 2e-16         

# Kappa : 0.8837          

# Mcnemar's Test P-Value : 0.07364         
                                          
#             Sensitivity : 0.8837          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.8958          
#              Prevalence : 0.5000          
#          Detection Rate : 0.4419          
#    Detection Prevalence : 0.4419          
#       Balanced Accuracy : 0.9419
# ```



# Next we check the performance of the trained model (fit_backstep1). Here we use the threshold $p*=0.5$.
fit_backstep1 <- glm(formula = V31 ~ V1 + V2 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + 
                       V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20 + V21 + V29 + 
                       V30, family = binomial, data = train, control = list(maxit = 50))
# ```{r,eval=FALSE}
# Predict the output from the model
a_backstep1 <- predict(fit_backstep1,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_backstep1 <- ifelse(a_backstep1>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_backstep1),as.factor(train$V31))
#           Reference
# Prediction  0  1
#          0 42  1
#          1  1 42

Accuracy : 1          
# 95% CI : (0.9924, 1)
# No Information Rate : 0.6501     
# P-Value [Acc > NIR] : < 2.2e-16  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         
                                     
#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.3499     
#          Detection Rate : 0.3499     
#    Detection Prevalence : 0.3499     
#       Balanced Accuracy : 1.0000
# ```



# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_backstep1 <- predict(fit_backstep1,newdata=test,type="response")
n_backstep1 <- ifelse(m_backstep1>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_backstep1),as.factor(test$V31))
# Accuracy : 0.9767          
# 95% CI : (0.9185, 0.9972)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9535          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 0.9767          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9767          
#          Neg Pred Value : 0.9767          
#              Prevalence : 0.5000          
#          Detection Rate : 0.4884          
#    Detection Prevalence : 0.5000          
#       Balanced Accuracy : 0.9767
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.6). Again, we use the threshold $p*=0.6$.
# ```{r,eval=FALSE}

m_backstep1 <- predict(fit_backstep1,newdata=test,type="response")
n_backstep1 <- ifelse(m_backstep1>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_backstep1),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 42  1
#          1  1 42

# Accuracy : 0.9767          
# 95% CI : (0.9185, 0.9972)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9535          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 0.9767          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9767          
#          Neg Pred Value : 0.9767          
#              Prevalence : 0.5000          
#          Detection Rate : 0.4884          
#    Detection Prevalence : 0.5000          
#       Balanced Accuracy : 0.9767
# ```




# Next we check the performance of the trained model (fit_lasso1). Here we use the threshold $p*=0.5$.
fit_lasso1 <- glm(formula = V31 ~ V8 + V10 + V11 + V12 + V15 + V16 + V20 + V21 + V22 + V25 + V27 + V28 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
# ```{r,eval=FALSE}
# Predict the output from the model
a_lasso1 <- predict(fit_lasso1,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_lasso1 <- ifelse(a_lasso1>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_lasso1),as.factor(train$V31))
#           Reference
# Prediction   0   1
#          0 165   1
#          1   4 313

# Accuracy : 0.9896         
# 95% CI : (0.976, 0.9966)
# No Information Rate : 0.6501         
# P-Value [Acc > NIR] : <2e-16         

# Kappa : 0.9772         

# Mcnemar's Test P-Value : 0.3711         
                                         
#             Sensitivity : 0.9763         
#             Specificity : 0.9968         
#          Pos Pred Value : 0.9940         
#          Neg Pred Value : 0.9874         
#              Prevalence : 0.3499         
#          Detection Rate : 0.3416         
#    Detection Prevalence : 0.3437         
#     Balanced Accuracy : 0.9866
# ```



# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_lasso1 <- predict(fit_lasso1,newdata=test,type="response")
n_lasso1 <- ifelse(m_lasso1>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso1),as.factor(test$V31))
#           Reference
# Prediction  0  1
#          0 43  1
#          1  0 42

# Accuracy : 0.9884          
# 95% CI : (0.9369, 0.9997)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9767          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9773          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#        Detection Rate : 0.5000          
#    Detection Prevalence : 0.5116          
#       Balanced Accuracy : 0.9884 
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.6). Again, we use the threshold $p*=0.6$.
# ```{r,eval=FALSE}

m_lasso1 <- predict(fit_lasso1,newdata=test,type="response")
n_lasso1 <- ifelse(m_lasso1>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso1),as.factor(test$V31))
#          Reference
# Prediction   0   1
#          0 165   2
#          1   4 312

# Accuracy : 0.9876          
# 95% CI : (0.9732, 0.9954)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9726          

# Mcnemar's Test P-Value : 0.6831          
                                          
#             Sensitivity : 0.9763          
#             Specificity : 0.9936          
#          Pos Pred Value : 0.9880          
#          Neg Pred Value : 0.9873          
#              Prevalence : 0.3499          
#          Detection Rate : 0.3416          
#    Detection Prevalence : 0.3458          
#       Balanced Accuracy : 0.9850
# ```




# Next we check the performance of the trained model (fit_lasso5). Here we use the threshold $p*=0.5$.
fit_lasso5 <- glm(formula = V31 ~ V8 + V11 + V12 + V15 + V16 + V21 + V22 + V27 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
# ```{r,eval=FALSE}
# Predict the output from the model
a_lasso5 <- predict(fit_lasso5,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_lasso5 <- ifelse(a_lasso5>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_lasso5),as.factor(train$V31))
#          Reference
# Prediction   0   1
#          0 165   2
#          1   4 312

# Accuracy : 0.9876          
# 95% CI : (0.9732, 0.9954)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9726          

# Mcnemar's Test P-Value : 0.6831          
                                          
#             Sensitivity : 0.9763          
#             Specificity : 0.9936          
#          Pos Pred Value : 0.9880          
#          Neg Pred Value : 0.9873          
#              Prevalence : 0.3499          
#          Detection Rate : 0.3416          
#    Detection Prevalence : 0.3458          
#       Balanced Accuracy : 0.9850 
# ```



# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_lasso5 <- predict(fit_lasso5,newdata=test,type="response")
n_lasso5 <- ifelse(m_lasso5>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso5),as.factor(test$V31))
#          Reference
# Prediction  0  1
#          0 43  1
#          1  0 42

# Accuracy : 0.9884          
# 95% CI : (0.9369, 0.9997)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9767          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9773          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5116          
#       Balanced Accuracy : 0.9884  
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.6). Again, we use the threshold $p*=0.6$.
# ```{r,eval=FALSE}

m_lasso5 <- predict(fit_lasso5,newdata=test,type="response")
n_lasso5 <- ifelse(m_lasso5>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso5),as.factor(test$V31))
#          Reference
# Prediction  0  1
#          0 43  1
#          1  0 42

# Accuracy : 0.9884          
# 95% CI : (0.9369, 0.9997)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9767          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9773          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5116          
#       Balanced Accuracy : 0.9884  
# ```




# Next we check the performance of the trained model (fit_lasso6). Here we use the threshold $p*=0.5$.
fit_lasso6 <- glm(formula = V31 ~ V8 + V11 + V15 + V16 + V21 + V22 + V27 + V29, 
                  family = binomial, data = train, control = list(maxit = 50))
# ```{r,eval=FALSE}
# Predict the output from the model
a_lasso6 <- predict(fit_lasso6,newdata=train,type="response")
# Set response >0.5 as 1 and <=0.5 as 0
b_lasso6 <- ifelse(a_lasso6>0.5, 1, 0)

# Compute the confusion matrix for training data
confusionMatrix(as.factor(b_lasso6),as.factor(train$V31))
#          Reference
# Prediction   0   1
#          0 165   3
#          1   4 311

# Accuracy : 0.9855          
# 95% CI : (0.9704, 0.9942)
# No Information Rate : 0.6501          
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9681          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 0.9763          
#             Specificity : 0.9904          
#          Pos Pred Value : 0.9821          
#          Neg Pred Value : 0.9873          
#              Prevalence : 0.3499          
#          Detection Rate : 0.3416          
#    Detection Prevalence : 0.3478          
#       Balanced Accuracy : 0.9834  
# ```



# Finally we use the trained model to make predictions for the test dataset (p=0.5). Again, we use the threshold $p*=0.5$.
# ```{r,eval=FALSE}

m_lasso6 <- predict(fit_lasso6,newdata=test,type="response")
n_lasso6 <- ifelse(m_lasso6>0.5, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso6),as.factor(test$V31))
#          Reference
# Prediction  0  1
#          0 43  1
#          1  0 42

# Accuracy : 0.9884          
# 95% CI : (0.9369, 0.9997)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9767          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9773          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5116          
#       Balanced Accuracy : 0.9884  
# ```


# Finally we use the trained model to make predictions for the test dataset (p=0.6). Again, we use the threshold $p*=0.6$.
# ```{r,eval=FALSE}

m_lasso6 <- predict(fit_lasso6,newdata=test,type="response")
n_lasso6 <- ifelse(m_lasso6>0.6, 1, 0)
# Compute the confusion matrix for test output
confusionMatrix(as.factor(n_lasso6),as.factor(test$V31))
#         Reference
# Prediction  0  1
#          0 43  1
#          1  0 42

# Accuracy : 0.9884          
# 95% CI : (0.9369, 0.9997)
# No Information Rate : 0.5             
# P-Value [Acc > NIR] : <2e-16          

# Kappa : 0.9767          

# Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 1.0000          
#             Specificity : 0.9767          
#          Pos Pred Value : 0.9773          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.5000          
#          Detection Rate : 0.5000          
#    Detection Prevalence : 0.5116          
#       Balanced Accuracy : 0.9884  
# ```






