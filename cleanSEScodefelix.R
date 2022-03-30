library(ggplot2)
library(MLmetrics)

#import data
data <- read.csv(file="C:\\Users\\s166646\\Downloads\\seh_data.csv",sep = ';')

#transform date column into compatible format for R
data$day <- as.Date(data$day, format="%Y-%m-%d")

#introduce prevalence in data
data$prevalence <- c(data$daily_cases/data$daily_tot)
summary(data$prevalence)

#plot the prevalence
plot(data$day, data$prevalence, ylim=c(-0.025, 0.60), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')

#perform SES using pre-existing library to check results later
ses.prevalence <- ses(data$prevalence, alpha=.2, h=50)
plot(ses.prevalence,ylim=c(-0.025, 0.20), main="SES Prevalence using pre-existing library", xlab="date", ylab="prevalence")
summary(ses.prevalence)

#perform SES using component form
componentSES <- function(alpha) {
  #Set y0 = 0
  predictedvals = c(0,0)
  for (i in 1:(length(data$prevalence))-1){
    et = data$prevalence[i] - predictedvals[i-1]
    predictedval = predictedvals[i-1] + alpha*et
    predictedvals <- append(predictedvals, predictedval)
  }
  #Weird interaction, but an extra 0 at the beginning of the vector is needed to produce correct results
  #predictedvals <- c(rep(0,1),predictedvals)
  return(predictedvals)
}

#SES in weighted average form
#Initially used to verify results, not used anymore as the code is inefficient and the component form is easier to expand on
waSESprevalence <- function(a) {
  prevalencecopy = data$prevalence
  daycopy = data$day
  #a = 2/(N+1)
  predictedvals = c(0) #put l0 to zero
  nrofiters = length(data$daily_tot) #predict 50 days into the future
  
  #Loop over all T's
  for (val in 1:nrofiters){
    allterms = 0
    j = 0
    
    #Sum from j=0 to T-1
    for (i in 0:(val-1)){
      term = a*((1-a)^j)*prevalencecopy[val-j]
      allterms = allterms + term
      j = j + 1
    } 
    predictedvals = append(predictedvals, allterms)
    
    #If there is no existing datapoint, add the prediction to the list
    if (val >= length(data[2])){
      prevalencecopy = append(prevalencecopy, allterms)
      
    }
  }
  
  return(predictedvals)
}

#Use both methods and compare results
waSESvals <- waSESprevalence(0.1)
componentSESvals <- componentSES(0.1)

#plot results
plot(componentSESvals,type='l',ylim=c(-0.025, 0.60), main="SES prevalence using component form", xlab="day", ylab="prevalence")
plot(waSESvals, ylim=c(-0.025, 0.60), type='l', main="SES prevalence using weighted average form", xlab="day", ylab="prevalence")

#Function to determine optimal alpha from a list of alphas
determineAlpha <- function(alphas){
  allMSEs <- c()
  for (alpha in alphas){
    
    predictions = componentSES(alpha)
    TotalMSE = MSE(predictions, data$prevalence)
    
    allMSEs <- append(allMSEs, TotalMSE)
  }
  plot(alphas,allMSEs, main='MSE per alpha value')
  
  lowestMSE = min(allMSEs)
  lowestMSEindex = which.min(allMSEs)
  lowestalpha = alphas[lowestMSEindex]
  
  print('lowest MSE:')
  print(lowestMSE)
  print('optimal alpha:')
  print(lowestalpha)
 
  return(lowestalpha)
}

determineAlpha(seq(0.15, 1, by=0.005))
