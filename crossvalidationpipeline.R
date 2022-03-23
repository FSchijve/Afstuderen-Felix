library(forecast)
library(MLmetrics)

#import data
data <- read.csv(file="C:\\Users\\s166646\\Downloads\\seh_data.csv",sep = ';')

#transform date column into compatible format for R
data$day <- as.Date(data$day, format="%Y-%m-%d")

#introduce prevalence in data
data$prevalence <- c(data$daily_cases/data$daily_tot)
summary(data$prevalence)

#plot the prevalence
plot(data$day, data$prevalence, ylim=c(-0.025, 0.70), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')
plot(data$day, data$daily_tot, ylim=c(-0.025, 50), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')
plot(data$day, data$daily_cases, ylim=c(-0.025, 50), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')

#SES in component form for training data
componentSEStraining <- function(alpha) {
  #Set y0 = 0
  predictedvals = c(0)
  for (i in 1:length(traindata$prevalence)){
    et = traindata$prevalence[i] - predictedvals[i-1]
    predictedval = predictedvals[i-1] + alpha*et
    predictedvals <- append(predictedvals, predictedval)
  }
  #Weird interaction, but an extra 0 at the beginning of the vector is needed to produce correct results
  predictedvals <- c(rep(0,1),predictedvals)
  return(predictedvals)
}

#SES in component form for testing data
componentSEStesting <- function(alpha) {
  #Set y0 = 0
  predictedvals = c(0)
  for (i in 1:length(testdata$prevalence)){
    et = testdata$prevalence[i] - predictedvals[i-1]
    predictedval = predictedvals[i-1] + alpha*et
    predictedvals <- append(predictedvals, predictedval)
  }
  #Weird interaction, but an extra 0 at the beginning of the vector is needed to produce correct results
  predictedvals <- c(rep(0,1),predictedvals)
  return(predictedvals)
}

#Function to determine optimal alpha from a list of alphas
determineAlpha <- function(alphas){
  allMSEs <- c()
  for (alpha in alphas){
    
    predictions = componentSEStraining(alpha)
    TotalMSE = MSE(predictions, data$prevalence)
    
    allMSEs <- append(allMSEs, TotalMSE)
  }
  plot(alphas,allMSEs, main='MSE per alpha value')
  
  lowestMSE = min(allMSEs)
  lowestMSEindex = which.min(allMSEs)
  lowestalpha = alphas[lowestMSEindex]
  
  print('lowest MSE:')
  print(lowestMSE)
  print('lowest alpha:')
  print(lowestalpha)
  
  return(lowestalpha)
}

#Create 10 folds
folds <- cut(seq(1,nrow(data)),breaks=nroffolds,labels=FALSE)

meanmse = 0
nroffolds = 5

for(i in 1:nroffolds){
  #Assign folds to training and testing sets
  testindexes <- which(folds==i,arr.ind=TRUE)
  trainindexes <- which(folds>i, arr.ind = TRUE)
  trainindexes <- append(trainindexes, which(folds<i, arr.ind = TRUE))
  testdata <- data[testindexes, ]
  traindata <- data[trainindexes, ]
  
  #perform model creation on training data
  bestalpha = determineAlpha(seq(0.01, 1, by=0.01))
  trainingpredictions = componentSEStraining(bestalpha)
  
  #Use ideal alpha to build model and perform on test set
  testpredictions = componentSEStesting(bestalpha)
  
  #print error on test set 
  mse = MSE(testpredictions, testdata$prevalence)
  print('MSE on test set:')
  print(mse)
  meanmse = meanmse + mse
}
meanmse = meanmse/nroffolds
print('mean MSE:')
print(meanmse)

crossvalidation <- function(nroffolds){
  #Create 10 folds
  folds <- cut(seq(1,nrow(data)),breaks=nroffolds,labels=FALSE)
  
  meanmse = 0
  
  for(i in 1:nroffolds){
    #Assign folds to training and testing sets
    testindexes <- which(folds==i,arr.ind=TRUE)
    trainindexes <- which(folds>i, arr.ind = TRUE)
    trainindexes <- append(trainindexes, which(folds<i, arr.ind = TRUE))
    testdata <- data[testindexes, ]
    traindata <- data[trainindexes, ]
    
    #perform model creation on training data
    bestalpha = determineAlpha(seq(0.01, 1, by=0.01))
    trainingpredictions = componentSEStraining(bestalpha)
    plot(traindata$prevalence)
    plot(trainingpredictions)
    
    #Use ideal alpha to build model and perform on test set
    testpredictions = componentSEStesting(bestalpha)
    
    #print error on test set 
    print('fold nr:')
    print(i)
    mse = MSE(testpredictions, testdata$prevalence)
    print('MSE on test set:')
    print(mse)
    meanmse = meanmse + mse
  }
  meanmse = meanmse/nroffolds
  print('total mean MSE:')
  print(meanmse)
}

crossvalidation(2)
#arimamodel <- auto.arima(data$prevalence, trace=TRUE)
#arimamodel.summary()
#predict(arimamodel,n.ahead=5)
#plot(forecast(arimamodel),h=5)
#summary(arimamodel)
#newarima <- arima(data$prevalence, order=c(1,0,1))
#plot(residuals(newarima))
#plot(residuals(arimamodel))

#Create 10 folds

nroffolds = 10
folds <- cut(seq(1,nrow(data)),breaks=nroffolds,labels=FALSE)

meanmse = 0

for(i in 1:nroffolds){
  #Assign folds to training and testing sets
  testindexes <- which(folds==i,arr.ind=TRUE)
  trainindexes <- which(folds>i, arr.ind = TRUE)
  trainindexes <- append(trainindexes, which(folds<i, arr.ind = TRUE))
  testdata <- data[testindexes, ]
  traindata <- data[trainindexes, ]
  
  #perform model creation on training data
  bestalpha = determineAlpha(seq(0.01, 1, by=0.01))
  trainingpredictions = componentSEStraining(bestalpha)
  plot(traindata$prevalence)
  plot(trainingpredictions)
  
  #Use ideal alpha to build model and perform on test set
  testpredictions = componentSEStesting(bestalpha)
  
  #print error on test set 
  print('fold nr:')
  print(i)
  mse = MSE(testpredictions, testdata$prevalence)
  print('MSE on test set:')
  print(mse)
  meanmse = meanmse + mse
}
meanmse = meanmse/nroffolds
print('total mean MSE:')
print(meanmse)