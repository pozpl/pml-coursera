library('caret')
library(pROC)
library(doMC)
library(Amelia)
registerDoMC(cores=4)
getDoParWorkers()

# load data
trainRawData <- read.csv("data/pml-training.csv",na.strings=c("NA",""))

missmap(trainRawData, main = "Missingness map for full training dataset")

# make training set
trainDataFull <-trainRawData[ , !apply(trainRawData, 2, function(x) any(is.na(x)) ) ]
trainDataFull$classe <- as.factor(trainDataFull$classe)

# remove unuseful columns
removeIndex<- grep("timestamp|X|user_name|new_window|num_window", colnames(trainDataFull))
trainDataCleaned <- trainDataFull[,-removeIndex]

trainIndex <- createDataPartition(y = trainDataCleaned$classe, p=0.007,list=FALSE)
trainData <- trainDataCleaned[trainIndex,]
testTrainData <- trainDataCleaned[-trainIndex,]

set.seed(6578)
# make RF model
modRFFit <- train(trainData$classe ~.,data = trainData,method="rf")
modRFFit

#Try to propose alternative model
fitControl <- trainControl(## 10-fold CV
    method = "cv",
    number = 10,
    ## repeated ten times
    ## repeats = 10
    )
gbmGrid <-  expand.grid(interaction.depth = 1,
                        n.trees = 200,
                        shrinkage = 0.1)
modGBM <- train(classe ~. , data = trainData, method = "gbm", trControl = fitControl, verbose = FALSE, tuneGrid = gbmGrid)

modGBM
#crosvalidation of the model
# answers
answersRFFit <- predict(modRFFit, testTrainData)
answersGbmFit <- predict(modGBM, testTrainData)

rfConfMatr <- confusionMatrix(testTrainData$classe, answersRFFit)
gbmConfMatr <- confusionMatrix(testTrainData$classe, answersGbmFit)

rfAccuracy <- rfConfMatr$overall['Accuracy']
gbmAccuracy <- gbmConfMatr$overall['Accuracy']
rfAccuracy
gbmAccuracy
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

answersRFFit
answersGbmFit
#source("mpl_write_file.R");
#pml_write_files(answersRFFit)
