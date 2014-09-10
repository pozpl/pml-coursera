library('caret')
library(pROC)
library(doMC)
registerDoMC(cores=3)
getDoParWorkers()

# load data
trainRawData <- read.csv("data/pml-training.csv",na.strings=c("NA",""))

# make training set
trainDataFull <-trainRawData[ , !apply(trainRawData, 2, function(x) any(is.na(x)) ) ]

# remove unuseful columns
removeIndex<- grep("timestamp|X|user_name|new_window|num_window", colnames(trainDataFull))
trainDataCleaned <- trainDataFull[,-removeIndex]

trainIndex <- createDataPartition(y = trainDataCleaned$classe, p=0.7,list=FALSE)
trainData <- trainDataCleaned[trainIndex,]
testTrainData <- trainDataCleaned[-trainIndex,]

set.seed(6578)
# make RF model
modRFFit <- train(trainData$classe ~.,data = trainData,method="rf")
modRFFit

#Try to propose alternative model
fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 10)
modGBM <- train(trainData$classe ~. ,
                data = trainData,
                method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                importance = TRUE)

modGBM
#crosvalidation of the model
# answers
answersRFFit <- predict(modRFFit, testTrainData)
answersGbmFit <- predict(modGBM, testTrainData)

confusionMatrix(testTrainData$classe, answersRFFit)
confusionMatrix(testTrainData$classe, answersGbmFit)


####Train winner model on all available data
modRFFit <- train(classe ~.,data = trainDataCleaned,method="rf")
modRFFit

# load test data
testRawData <- read.csv("data/pml-testing.csv",na.strings=c("NA",""))
testData<-testRawData[ , !apply(testRawData, 2, function(x) any(is.na(x)) ) ]
testData <- testData[,-removeIndex]

# answers
answersRFFit <- predict(modRFFit, testData,type='raw')
answersGbmFit <- predict(modGBM, testData,type='raw')

source("mpl_write_file.R");
pml_write_files(answersRFFit)
