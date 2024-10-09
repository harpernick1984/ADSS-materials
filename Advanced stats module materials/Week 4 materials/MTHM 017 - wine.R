library(C50)
library(kernlab) 
library(randomForest)
library(caret)

# Import the wine dataset
wine <- read.csv('wine.csv')
y <- wine[,1]
x <- wine[,-1]


## Try with p = 0.7

# Split data into Train and Test dataset
ii <- createDataPartition(y, p=.7, list=F) ## returns indices for train data 
xTrain <- x[ii,]; yTrain <- y[ii]
xTest <- x[-ii,]; yTest <- y[-ii]


## Decision Tree ##

# Fit decision tree
mdl <- C5.0(x=xTrain, y=yTrain) 

# Predict on test dataset
yTestPred <- predict(mdl, newdata=xTest) 
confusionMatrix(yTestPred, yTest)
# Reference
# Prediction  A  B  C
#          A 17  2  0
#          B  0 17  0
#          C  0  2 14

# Overall Statistics

# Accuracy : 0.9231          
# 95% CI : (0.8146, 0.9786)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : 4.536e-15       

# Kappa : 0.8844          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.8095   1.0000
# Specificity            0.9429   1.0000   0.9474
# Pos Pred Value         0.8947   1.0000   0.8750
# Neg Pred Value         1.0000   0.8857   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3269   0.2692
# Detection Prevalence   0.3654   0.3269   0.3077
# Balanced Accuracy      0.9714   0.9048   0.9737


## Random Forest ##

# Fit Random Forest model
# Fix ntree and mtry
set.seed(1040) # for reproducibility 
mdl <- train(x=xTrain, y=yTrain,
             method='rf',
             ntree=200, 
             tuneGrid=data.frame(mtry=2))

yTestPred <- predict(mdl, newdata=xTest) 
confusionMatrix(yTestPred, yTest)
# Reference
# Prediction  A  B  C
#          A 17  1  0
#          B  0 20  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 0.9808          
# 95% CI : (0.8974, 0.9995)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9708          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9524   1.0000
# Specificity            0.9714   1.0000   1.0000
# Pos Pred Value         0.9444   1.0000   1.0000
# Neg Pred Value         1.0000   0.9687   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3846   0.2692
# Detection Prevalence   0.3462   0.3846   0.2692
# Balanced Accuracy      0.9857   0.9762   1.0000


## SVM with linear kernel ##

mdl <- train(x=xTrain,y=yTrain, method='svmLinear') 
print(mdl)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results:
  
#   Accuracy   Kappa    
# 0.9619321  0.9419248

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
# Reference
# Prediction  A  B  C
#          A 17  0  0
#          B  0 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9808          
# 95% CI : (0.8974, 0.9995)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9709          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9524   1.0000
# Specificity            1.0000   1.0000   0.9737
# Pos Pred Value         1.0000   1.0000   0.9333
# Neg Pred Value         1.0000   0.9687   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3846   0.2692
# Detection Prevalence   0.3269   0.3846   0.2885
# Balanced Accuracy      1.0000   0.9762   0.9868

mdl <- train(x=xTrain,y=yTrain, method = "svmLinear",
              trControl = trainControl("cv", number = 5),
              tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl)

# Print the best tuning parameter C that maximises model accuracy
mdl$bestTune
#           C
# 3 0.2105263




## SVM with radial kernel ##

mdl <- train(x=xTrain,y=yTrain, method='svmRadial') 
print(mdl)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results across tuning parameters:
  
#   C     Accuracy   Kappa    
# 0.25  0.9607155  0.9391947
# 0.50  0.9634717  0.9435381
# 1.00  0.9626771  0.9421745

# Tuning parameter 'sigma' was held constant at a value of 0.06774281
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 0.06774281 and C = 0.5.

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 17  0  0
#          B  0 21  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 1          
# 95% CI : (0.9315, 1)
# No Information Rate : 0.4038     
# P-Value [Acc > NIR] : < 2.2e-16  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   1.0000   1.0000
# Specificity            1.0000   1.0000   1.0000
# Pos Pred Value         1.0000   1.0000   1.0000
# Neg Pred Value         1.0000   1.0000   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.4038   0.2692
# Detection Prevalence   0.3269   0.4038   0.2692
# Balanced Accuracy      1.0000   1.0000   1.0000


## SVM with polynomial kernel ##

mdl <- train(x=xTrain,y=yTrain, method='svmPoly') 
print(mdl)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results across tuning parameters:
  
#   degree  scale  C     Accuracy   Kappa      
# 1       0.001  0.25  0.3686787  0.000000000
# 1       0.001  0.50  0.3686787  0.000000000
# 1       0.001  1.00  0.3739389  0.009824711
# 1       0.010  0.25  0.8428859  0.761933533
# 1       0.010  0.50  0.9648293  0.946167659
# 1       0.010  1.00  0.9674415  0.950352284
# 1       0.100  0.25  0.9753780  0.962366769
# 1       0.100  0.50  0.9795258  0.968666921
# 1       0.100  1.00  0.9719702  0.957141401
# 2       0.001  0.25  0.3686787  0.000000000
# 2       0.001  0.50  0.3739389  0.009824711
# 2       0.001  1.00  0.6980659  0.539086624
# 2       0.010  0.25  0.9648996  0.946301350
# 2       0.010  0.50  0.9709656  0.955636638
# 2       0.010  1.00  0.9755726  0.962675663
# 2       0.100  0.25  0.9753514  0.962190198
# 2       0.100  0.50  0.9739207  0.960124782
# 2       0.100  1.00  0.9722325  0.957481158
# 3       0.001  0.25  0.3686787  0.000000000
# 3       0.001  0.50  0.5034248  0.229473960
# 3       0.001  1.00  0.9171717  0.873958800
# 3       0.010  0.25  0.9693506  0.953094033
# 3       0.010  0.50  0.9719735  0.957197305
# 3       0.010  1.00  0.9813340  0.971381741
# 3       0.100  0.25  0.9717510  0.956797336
# 3       0.100  0.50  0.9709026  0.955419070
# 3       0.100  1.00  0.9699270  0.953946907

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were degree = 3, scale = 0.01 and C = 1.

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 17  1  0
#          B  0 20  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 0.9808          
# 95% CI : (0.8974, 0.9995)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9708          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9524   1.0000
# Specificity            0.9714   1.0000   1.0000
# Pos Pred Value         0.9444   1.0000   1.0000
# Neg Pred Value         1.0000   0.9687   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3846   0.2692
# Detection Prevalence   0.3462   0.3846   0.2692
# Balanced Accuracy      0.9857   0.9762   1.0000



## Retry with p = 0.07

# Split data into Train and Test dataset
ii2 <- createDataPartition(y, p=.07, list=F) ## returns indices for train data 
xTrain2 <- x[ii,]; yTrain2 <- y[ii]
xTest2 <- x[-ii,]; yTest2 <- y[-ii]


## Decision Tree ##

# Fit decision tree
mdl2 <- C5.0(x=xTrain2, y=yTrain2) 

# Predict on test dataset
yTestPred2 <- predict(mdl2, newdata=xTest2) 
confusionMatrix(yTestPred2, yTest2)
#           Reference
# Prediction  A  B  C
#          A 17  2  0
#          B  0 17  0
#          C  0  2 14

# Overall Statistics

# Accuracy : 0.9231          
# 95% CI : (0.8146, 0.9786)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : 4.536e-15       

# Kappa : 0.8844          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.8095   1.0000
# Specificity            0.9429   1.0000   0.9474
# Pos Pred Value         0.8947   1.0000   0.8750
# Neg Pred Value         1.0000   0.8857   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3269   0.2692
# Detection Prevalence   0.3654   0.3269   0.3077
# Balanced Accuracy      0.9714   0.9048   0.9737


## Random Forest ##

# Fit Random Forest model
# Fix ntree and mtry
set.seed(1040) # for reproducibility 
mdl2 <- train(x=xTrain2, y=yTrain2,
             method='rf',
             ntree=200, 
             tuneGrid=data.frame(mtry=2))

yTestPred2 <- predict(mdl2, newdata=xTest2) 
confusionMatrix(yTestPred2, yTest2)
#           Reference
# Prediction  A  B  C
#          A 17  1  0
#          B  0 20  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 0.9808          
# 95% CI : (0.8974, 0.9995)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9708          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9524   1.0000
# Specificity            0.9714   1.0000   1.0000
# Pos Pred Value         0.9444   1.0000   1.0000
# Neg Pred Value         1.0000   0.9687   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3846   0.2692
# Detection Prevalence   0.3462   0.3846   0.2692
# Balanced Accuracy      0.9857   0.9762   1.0000


## SVM with linear kernel ##

mdl2 <- train(x=xTrain2,y=yTrain2, method='svmLinear') 
print(mdl2)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results:
  
#   Accuracy  Kappa    
# 0.964447  0.9450548

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
# Reference
# Prediction  A  B  C
#          A 17  0  0
#          B  0 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9808          
# 95% CI : (0.8974, 0.9995)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9709          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9524   1.0000
# Specificity            1.0000   1.0000   0.9737
# Pos Pred Value         1.0000   1.0000   0.9333
# Neg Pred Value         1.0000   0.9687   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3846   0.2692
# Detection Prevalence   0.3269   0.3846   0.2885
# Balanced Accuracy      1.0000   0.9762   0.9868

mdl2 <- train(x=xTrain,y=yTrain, method = "svmLinear",
              trControl = trainControl("cv", number = 5),
              tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl2)

# Print the best tuning parameter C that maximises model accuracy
mdl2$bestTune
#           C
# 4 0.3157895



## SVM with radial kernel ##

mdl2 <- train(x=xTrain2,y=yTrain2, method='svmRadial') 
print(mdl2)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results across tuning parameters:
  
#   C     Accuracy   Kappa    
# 0.25  0.9751995  0.9620750
# 0.50  0.9771263  0.9650274
# 1.00  0.9772499  0.9651270

# Tuning parameter 'sigma' was held constant at a value of 0.05731695
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 0.05731695 and C = 1.

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 17  0  0
#          B  0 21  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 1          
# 95% CI : (0.9315, 1)
# No Information Rate : 0.4038     
# P-Value [Acc > NIR] : < 2.2e-16  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   1.0000   1.0000
# Specificity            1.0000   1.0000   1.0000
# Pos Pred Value         1.0000   1.0000   1.0000
# Neg Pred Value         1.0000   1.0000   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.4038   0.2692
# Detection Prevalence   0.3269   0.4038   0.2692
# Balanced Accuracy      1.0000   1.0000   1.0000


## SVM with polynomial kernel ##

mdl2 <- train(x=xTrain2,y=yTrain2, method='svmPoly') 
print(mdl2)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results across tuning parameters:
  
#   degree  scale  C     Accuracy   Kappa     
# 1       0.001  0.25  0.3643790  0.02585635
# 1       0.001  0.50  0.3643790  0.02585635
# 1       0.001  1.00  0.3680154  0.03293118
# 1       0.010  0.25  0.8205887  0.73066610
# 1       0.010  0.50  0.9551520  0.93133938
# 1       0.010  1.00  0.9581515  0.93587108
# 1       0.100  0.25  0.9620083  0.94161175
# 1       0.100  0.50  0.9696653  0.95360042
# 1       0.100  1.00  0.9718622  0.95674761
# 2       0.001  0.25  0.3643790  0.02585635
# 2       0.001  0.50  0.3680154  0.03293118
# 2       0.001  1.00  0.6819894  0.51834778
# 2       0.010  0.25  0.9508103  0.92467883
# 2       0.010  0.50  0.9589359  0.93693833
# 2       0.010  1.00  0.9600656  0.93857077
# 2       0.100  0.25  0.9676953  0.95063789
# 2       0.100  0.50  0.9676077  0.95031097
# 2       0.100  1.00  0.9655891  0.94717436
# 3       0.001  0.25  0.3633790  0.02400000
# 3       0.001  0.50  0.4915769  0.22919963
# 3       0.001  1.00  0.8968514  0.84457084
# 3       0.010  0.25  0.9585891  0.93646498
# 3       0.010  0.50  0.9627116  0.94278556
# 3       0.010  1.00  0.9692611  0.95282973
# 3       0.100  0.25  0.9701490  0.95407115
# 3       0.100  0.50  0.9701490  0.95410788
# 3       0.100  1.00  0.9701170  0.95407377

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were degree = 1, scale = 0.1 and C = 1.

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 17  2  0
#          B  0 19  0
#          C  0  0 14

# Overall Statistics

# Accuracy : 0.9615          
# 95% CI : (0.8679, 0.9953)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9418          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.9048   1.0000
# Specificity            0.9429   1.0000   1.0000
# Pos Pred Value         0.8947   1.0000   1.0000
# Neg Pred Value         1.0000   0.9394   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3654   0.2692
# Detection Prevalence   0.3654   0.3654   0.2692
# Balanced Accuracy      0.9714   0.9524   1.0000




