library(ggplot2)

#import data
data <- read.csv(file="C:\\Users\\s166646\\Downloads\\prevalentie_data.csv",sep = ';')

#transform date column into compatible format for R
data$day <- as.Date(data$day, format="%Y-%m-%d")

#introduce prevalence in data
data$prevalence <- c(data$daily_cases/data$daily_tot)
summary(data$prevalence)

#plot the prevalence
plot(data$day, data$prevalence, ylim=c(-0.025, 0.20), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')

#perform SES using pre-existing library to check results later
ses.prevalence <- ses(data$prevalence, alpha=.2, h=50)
plot(x=data$day, y=ses.prevalence,ylim=c(-0.025, 0.20), main="SES Prevalence using pre-existing library", xlab="date", ylab="prevalence")
summary(ses.prevalence)

#SES in component form: y[t+1] = a*y[t] + (1-a)*y[t-1]
componentSESprevalence <- function(a) {
  predictedvals = c(data$prevalence[1])
  for (i in 1:length(data$prevalence)){
    value = a*data$prevalence[i] + ((1-a)*data$prevalence[i-1])
    predictedvals <- append(predictedvals, value)
  }
  return(predictedvals)
}
componentSESvals <- componentSESprevalence(0.2)

componentSESprevalence2 <-function(a){
  
}

#plot the SES component form results
plot(x=data$day,y=componentSESvals,type='l',ylim=c(-0.025, 0.20), main="SES prevalence using component form", xlab="date", ylab="prevalence")

#Weighted average SES
waSES <- function(N) {
  #work with copies as the formula is gonna adjust some data
  prevalencecopy = data$prevalence
  daycopy = data$day
  
  #Define prediction and sliding window sizes
  a = 2/(N+1)
  predictedvals = c(0) #put l0 to zero
  nrofiters = length(data$prevalence)+50 #predict 50 days into the future
  
  #we need to predict every Y value from t=1 to t=507
  for (T in 1:nrofiters){
    
    #Sum from j=0 to j=T-1
    j <- (0:(T-1)) 
    allterms = sum(a*((1-a)^j)*prevalencecopy[T-j])
    predictedvals = append(predictedvals, allterms)
    
    #If there is no existing datapoint, add the prediction to the list
    if (T >= length(data[2])){
      prevalencecopy = append(prevalencecopy, allterms)
    }
  }
  return(predictedvals)
}

waSESvals <- waSES(50)
