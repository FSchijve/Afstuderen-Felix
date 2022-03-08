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

#SES in weighted average form
waSESprevalence <- function(N) {
  prevalencecopy = data$prevalence
  daycopy = data$day
  a = 2/(N+1)
  predictedvals = c(0) #put l0 to zero
  nrofiters = 507 #predict 50 days into the future
  
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
waSESvals <- waSESprevalence(50)

print(daycopy[length(daycopy)])

#plot the SES wa form results
#because we are predicting into the future, the date list should be updated as well
daycopy <- data$day
difference <- length(waSESvals)-length(daycopy)

daycopy <- append(daycopy,daycopy[length(daycopy)]+1)
plot(x=data$day,y=waSESvals,type='l',ylim=c(-0.025, 0.20), main="SES prevalence using weighted average form", xlab="date", ylab="prevalence")