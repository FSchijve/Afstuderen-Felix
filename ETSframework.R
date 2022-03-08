#import data
data <- read.csv(file="C:\\Users\\s166646\\Downloads\\prevalentie_data.csv",sep = ';')

#transform date column into compatible format for R
data$day <- as.Date(data$day, format="%Y-%m-%d")

#introduce prevalence in data
data$prevalence <- c(data$daily_cases/data$daily_tot)
summary(data$prevalence)

#plot the prevalence
plot(data$day, data$prevalence, ylim=c(-0.025, 0.20), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')

#Weighted average SES
waSEStry <- function(N) {
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

#SES in weighted average form
waSESprevalence <- function(N) {
  prevalencecopy = data$prevalence
  daycopy = data$day
  a = 2/(N+1)
  predictedvals = c(0) #put l0 to zero
  nrofiters = length(data$prevalence)+50 #predict 50 days into the future
  
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

predictedvals = waSEStry(20)
predictedvals2 = waSESprevalence(20)

#Plot prediction values
plotpredictions <- function(predictedvals){
  daycopy <- data$day
  difference <- length(predictedvals)-length(daycopy)
  for (i in (1:difference)){
    daycopy <- append(daycopy,daycopy[length(daycopy)]+1)
  }
  plot(x=daycopy,y=predictedvals,type='l',ylim=c(-0.025, 0.20), xlab="date", ylab="prevalence")
  
}

#Check if new method works like the old, validated one
plotpredictions(predictedvals)
plotpredictions(predictedvals2)

#try forecast.ets function 
sesmodel = ets(y=data$prevalence, model="ANN",use.initial.values = FALSE, opt.crit="mse")

plot(sesmodel)
summary(sesmodel)

sesmodel = ets(y=data$prevalence, model="AMN",use.initial.values = FALSE, opt.crit="mse")

#Try all additive components of ETS framework
errortypes = c("A")
trendtypes = c("N","A","M")
seasonalitytypes = c("N","A","M")
#Obtain vector with all ETS types as strings
possiblecombinations = do.call(paste,c(expand.grid(errortypes,trendtypes,seasonalitytypes), list(sep='')))


# Possible to do this?
#apply(possiblecombinations, FUN = ets(data$prevalence, model = ))

for (combination in possiblecombinations){
  print(combination)
  modellist = c()
  model = ets(y=data$prevalence, model = combination, use.initial.values = FALSE, opt.crit="mse")
  
  modellist <- append(modellist, model)
}

