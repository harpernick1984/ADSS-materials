library(C50)
library(kernlab) 
library(randomForest)
library(caret)

# Import the gene dataset
gene <- read.csv('gene_expression.csv',row.names=1)

## Try with p = 0.7

# Extract labels as factors - B- or T-cell acute lymphocytic leukaemia
y <- as.factor(substr(colnames(gene),1,1))
x <- t(gene)
rownames(x) <- NULL

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
#           Reference
# Prediction  B  T
#          B 28  0
#          T  0  9

# Accuracy : 1          
# 95% CI : (0.9051, 1)
# No Information Rate : 0.7568     
# P-Value [Acc > NIR] : 3.322e-05  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         
                                     
#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.7568     
#          Detection Rate : 0.7568     
#    Detection Prevalence : 0.7568     
#       Balanced Accuracy : 1.0000     
                                     
#        'Positive' Class : B


## Random Forest ##

# Fit Random Forest model
# Fix ntree and mtry
set.seed(1040) # for reproducibility 
mdl <- train(x=xTrain, y=yTrain,
             method='rf',
             ntree=200, 
             tuneGrid=data.frame(mtry=100))

yTestPred <- predict(mdl, newdata=xTest) 
confusionMatrix(yTestPred, yTest)
#           Reference
# Prediction  B  T
#          B 28  0
#          T  0  9

# Accuracy : 1          
# 95% CI : (0.9051, 1)
# No Information Rate : 0.7568     
# P-Value [Acc > NIR] : 3.322e-05  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         

#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.7568     
#          Detection Rate : 0.7568     
#    Detection Prevalence : 0.7568     
#       Balanced Accuracy : 1.0000     

#        'Positive' Class : B

# Variable importance by mean decrease in gini index
varImp(mdl$finalModel)


## SVM with linear kernel ##

mdl <- train(x=xTrain,y=yTrain, method='svmLinear') 
print(mdl)
# 91 samples
# 12625 predictors
# 2 classes: 'B', 'T' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 91, 91, 91, 91, 91, 91, ... 
# Resampling results:

# Accuracy  Kappa
# 1         1 

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
#           Reference
# Prediction  B  T
#          B 28  0
#          T  0  9

# Accuracy : 1          
# 95% CI : (0.9051, 1)
# No Information Rate : 0.7568     
# P-Value [Acc > NIR] : 3.322e-05  

# Kappa : 1          

# Mcnemar's Test P-Value : NA         

#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.7568     
#          Detection Rate : 0.7568     
#    Detection Prevalence : 0.7568     
#       Balanced Accuracy : 1.0000     

#        'Positive' Class : B

mdl <- train(x=xTrain,y=yTrain, method = "svmLinear",
             trControl = trainControl("cv", number = 5),
             tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl)

# Print the best tuning parameter C that maximises model accuracy
mdl$bestTune
#           C
# 5 0.4210526



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
# 0.25  0.9810916  0.9708104
# 0.50  0.9810048  0.9708160
# 1.00  0.9800974  0.9693306

# Tuning parameter 'sigma' was held constant at a value of 0.0720873
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 0.0720873 and C = 0.25.

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
# Reference
# Prediction  A  B  C
#          A 16  0  0
#          B  1 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9615          
# 95% CI : (0.8679, 0.9953)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9416          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9524   1.0000
# Specificity            1.0000   0.9677   0.9737
# Pos Pred Value         1.0000   0.9524   0.9333
# Neg Pred Value         0.9722   0.9677   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3846   0.2692
# Detection Prevalence   0.3077   0.4038   0.2885
# Balanced Accuracy      0.9706   0.9601   0.9868

mdl <- train(x=xTrain,y=yTrain, method = "svmRadial",
             trControl = trainControl("cv", number = 5),
             tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl)

# Print the best tuning parameter C that maximises model accuracy
mdl$bestTune
#           C
# 5 0.4210526



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
# 1       0.001  0.25  0.3863871  0.04014670
# 1       0.001  0.50  0.3863871  0.04014670
# 1       0.001  1.00  0.4334018  0.11157001
# 1       0.010  0.25  0.9045274  0.85433858
# 1       0.010  0.50  0.9525055  0.92692427
# 1       0.010  1.00  0.9549492  0.93063443
# 1       0.100  0.25  0.9617644  0.94103399
# 1       0.100  0.50  0.9601964  0.93872124
# 1       0.100  1.00  0.9601915  0.93879905
# 2       0.001  0.25  0.3863871  0.04014670
# 2       0.001  0.50  0.4334018  0.11157001
# 2       0.001  1.00  0.8126953  0.71390444
# 2       0.010  0.25  0.9508407  0.92433550
# 2       0.010  0.50  0.9557335  0.93182691
# 2       0.010  1.00  0.9615366  0.94058704
# 2       0.100  0.25  0.9678728  0.95042121
# 2       0.100  0.50  0.9648674  0.94586649
# 2       0.100  1.00  0.9631571  0.94336275
# 3       0.001  0.25  0.3969695  0.05639401
# 3       0.001  0.50  0.6319404  0.42994016
# 3       0.001  1.00  0.9407112  0.90902864
# 3       0.010  0.25  0.9540491  0.92927530
# 3       0.010  0.50  0.9616532  0.94082720
# 3       0.010  1.00  0.9609573  0.93977763
# 3       0.100  0.25  0.9548607  0.93035900
# 3       0.100  0.50  0.9531258  0.92780770
# 3       0.100  1.00  0.9549651  0.93056185

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were degree = 2, scale = 0.1 and C = 0.25.

# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred, yTest) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 16  0  0
#          B  1 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9615          
# 95% CI : (0.8679, 0.9953)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9416          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9524   1.0000
# Specificity            1.0000   0.9677   0.9737
# Pos Pred Value         1.0000   0.9524   0.9333
# Neg Pred Value         0.9722   0.9677   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3846   0.2692
# Detection Prevalence   0.3077   0.4038   0.2885
# Balanced Accuracy      0.9706   0.9601   0.9868

mdl <- train(x=xTrain,y=yTrain, method = "svmPoly",
             trControl = trainControl("cv", number = 5),
             tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl)

# Print the best tuning parameter C that maximises model accuracy
mdl$bestTune
#           C
# 5 0.4210526




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
#          A 16  0  0
#          B  1 20  1
#          C  0  1 13

# Overall Statistics

# Accuracy : 0.9423          
# 95% CI : (0.8405, 0.9879)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : 2.471e-16       

# Kappa : 0.9121          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9524   0.9286
# Specificity            1.0000   0.9355   0.9737
# Pos Pred Value         1.0000   0.9091   0.9286
# Neg Pred Value         0.9722   0.9667   0.9737
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3846   0.2500
# Detection Prevalence   0.3077   0.4231   0.2692
# Balanced Accuracy      0.9706   0.9439   0.9511


## Random Forest ##

# Fit Random Forest model
# Fix ntree and mtry
set.seed(1040) # for reproducibility 
mdl2 <- train(x=xTrain2, y=yTrain2,
             method='rf',
             ntree=200, 
             tuneGrid=data.frame(mtry=100))

yTestPred2 <- predict(mdl2, newdata=xTest2) 
confusionMatrix(yTestPred2, yTest2)
#           Reference
# Prediction  A  B  C
#          A 16  0  0
#          B  1 19  0
#          C  0  2 14

# Overall Statistics

# Accuracy : 0.9423          
# 95% CI : (0.8405, 0.9879)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : 2.471e-16       

# Kappa : 0.9128          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9048   1.0000
# Specificity            1.0000   0.9677   0.9474
# Pos Pred Value         1.0000   0.9500   0.8750
# Neg Pred Value         0.9722   0.9375   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3654   0.2692
# Detection Prevalence   0.3077   0.3846   0.3077
# Balanced Accuracy      0.9706   0.9363   0.9737


## SVM with linear kernel ##

mdl2 <- train(x=xTrain,y=yTrain2, method='svmLinear') 
print(mdl2)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results:
  
#   Accuracy   Kappa    
#  0.9645031  0.9454986

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 17  1  0
#          B  0 18  0
#          C  0  2 14

# Overall Statistics

# Accuracy : 0.9423          
# 95% CI : (0.8405, 0.9879)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : 2.471e-16       

# Kappa : 0.9131          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            1.0000   0.8571   1.0000
# Specificity            0.9714   1.0000   0.9474
# Pos Pred Value         0.9444   1.0000   0.8750
# Neg Pred Value         1.0000   0.9118   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3269   0.3462   0.2692
# Detection Prevalence   0.3462   0.3462   0.3077
# Balanced Accuracy      0.9857   0.9286   0.9737

mdl2 <- train(x=xTrain,y=yTrain, method = "svmLinear",
             trControl = trainControl("cv", number = 5),
             tuneGrid = expand.grid(C = seq(0.0, 2, length = 20)))

# Plot model accuracy vs different values of Cost
plot(mdl2)

# Print the best tuning parameter C that maximises model accuracy
mdl2$bestTune
#           C
# 5 0.4210526



## SVM with radial kernel ##

mdl2 <- train(x=xTrain,y=yTrain, method='svmRadial') 
print(mdl2)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 126, 126, 126, 126, 126, 126, ... 
# Resampling results across tuning parameters:
  
#   C     Accuracy   Kappa    
# 0.25  0.9780498  0.9662985
# 0.50  0.9814690  0.9714808
# 1.00  0.9799719  0.9691609

# Tuning parameter 'sigma' was held constant at a value of 0.06384195
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 0.06384195 and C = 0.5.

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 16  0  0
#          B  1 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9615          
# 95% CI : (0.8679, 0.9953)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9416          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9524   1.0000
# Specificity            1.0000   0.9677   0.9737
# Pos Pred Value         1.0000   0.9524   0.9333
# Neg Pred Value         0.9722   0.9677   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3846   0.2692
# Detection Prevalence   0.3077   0.4038   0.2885
# Balanced Accuracy      0.9706   0.9601   0.9868


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
# 1       0.001  0.25  0.3414127  0.00000000
# 1       0.001  0.50  0.3414127  0.00000000
# 1       0.001  1.00  0.3589953  0.02454529
# 1       0.010  0.25  0.8825444  0.82371216
# 1       0.010  0.50  0.9544541  0.93058600
# 1       0.010  1.00  0.9629198  0.94325734
# 1       0.100  0.25  0.9725338  0.95789616
# 1       0.100  0.50  0.9748715  0.96160911
# 1       0.100  1.00  0.9722110  0.95746590
# 2       0.001  0.25  0.3414127  0.00000000
# 2       0.001  0.50  0.3589953  0.02454529
# 2       0.001  1.00  0.7635568  0.64618545
# 2       0.010  0.25  0.9526905  0.92792812
# 2       0.010  0.50  0.9606237  0.93979290
# 2       0.010  1.00  0.9707525  0.95515803
# 2       0.100  0.25  0.9695578  0.95341442
# 2       0.100  0.50  0.9620000  0.94192232
# 2       0.100  1.00  0.9612340  0.94072260
# 3       0.001  0.25  0.3414127  0.00000000
# 3       0.001  0.50  0.5488818  0.31750243
# 3       0.001  1.00  0.9215788  0.88087304
# 3       0.010  0.25  0.9577700  0.93549284
# 3       0.010  0.50  0.9674106  0.95010965
# 3       0.010  1.00  0.9759872  0.96325532
# 3       0.100  0.25  0.9560672  0.93300936
# 3       0.100  0.50  0.9560125  0.93288760
# 3       0.100  1.00  0.9560766  0.93295766

# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were degree = 3, scale = 0.01 and C = 1.

# Test model on testing data
yTestPred2 <- predict(mdl2, newdata=xTest2)
# yTestPred <- mdl %>% predict(xTest) 
confusionMatrix(yTestPred2, yTest2) # predicted/true
#           Reference
# Prediction  A  B  C
#          A 16  0  0
#          B  1 20  0
#          C  0  1 14

# Overall Statistics

# Accuracy : 0.9615          
# 95% CI : (0.8679, 0.9953)
# No Information Rate : 0.4038          
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9416          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C
# Sensitivity            0.9412   0.9524   1.0000
# Specificity            1.0000   0.9677   0.9737
# Pos Pred Value         1.0000   0.9524   0.9333
# Neg Pred Value         0.9722   0.9677   1.0000
# Prevalence             0.3269   0.4038   0.2692
# Detection Rate         0.3077   0.3846   0.2692
# Detection Prevalence   0.3077   0.4038   0.2885
# Balanced Accuracy      0.9706   0.9601   0.9868





# Set training options
# Repeat 5-fold cross-validation, ten times
opts <- trainControl(method='repeatedcv', number=5, repeats=10, p=0.7)

# Fit SVM with linear kernel
set.seed(1040) # for reproducibility
mdl <- train(x=xTrain, y=yTrain,            # training data 
             method='svmLinear',            # machine learning model
             trControl=opts,                # training options
             tuneGrid=data.frame(C=c(0.01, 1, 10, 100, 1000))) # range of C's to try
print(mdl)
# 126 samples
# 13 predictor
# 3 classes: 'A', 'B', 'C' 

# No pre-processing
# Resampling: Cross-Validated (5 fold, repeated 10 times) 
# Summary of sample sizes: 100, 102, 101, 101, 100, 100, ... 
# Resampling results across tuning parameters:
  
#   C      Accuracy   Kappa    
# 1e-02  0.9698359  0.9543645
# 1e+00  0.9745744  0.9615187
# 1e+01  0.9745744  0.9615187
# 1e+02  0.9745744  0.9615187
# 1e+03  0.9745744  0.9615187





Accuracy was used to select the optimal model using the largest value.
The final value used for the model was C = 1.


# Test model on testing data
yTestPred <- predict(mdl, newdata=xTest)
confusionMatrix(yTestPred, yTest) # predicted/true

