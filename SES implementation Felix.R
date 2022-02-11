library("readxl")

#Import data
data <- read_excel("C:\\Users\\s166646\\Downloads\\prevalentie_data.xlsx")
dates = data[[1]]
dailycases = data[[2]]
dailytot = data[[3]]

#Transform data towards daily prevalence
dailyprevalence = c()
for (i in 1:length(dailycases)){
  prevalence = dailycases[i]/dailytot[i]
  dailyprevalence = append(dailyprevalence,prevalence)
}

#Plot daily prevalence
plot(dailyprevalence,ylim=c(-0.025, 0.20), type='l', main='Daily Prevalence', xlab='days')

#Perform SES with premade library to compare results later
library(tidyverse)
library(fpp2)  
ses.dailyprevalence <- ses(dailyprevalence, alpha = .2, h = 100)
plot(ses.dailyprevalence,ylim=c(-0.025, 0.20), main="SES Prevalence using pre-existing library", xlab="days", ylab="prevalence")
summary(ses.dailyprevalence)

#SES in component form: y(t+1) = ay(t)+(1-a)y(t-1) 
y0 = c(dailyprevalence[1])
a = 0.2
for (val in 1:length(dailyprevalence)){
  y = a*dailyprevalence[val]  + ((1-a)*dailyprevalence[val-1]) 
  y0 <- append(y0, y)
}

#Plot results
plot(y0,type='l',ylim=c(-0.025, 0.20), main="SES prevalence using component form", xlab="days", ylab="prevalence")
summary(dailyprevalence)
summary(y0)

#SES in weighted average form
dailyprevalencecopy <- dailyprevalence #We create a copy because the forecasts of future daily prevalences are added to the daily prevalence list 
N=20
a= 2/(N+1)
#a = 0.09
yvals = c(0)
nrofiters = 557 #predict 100 days into the future

#loop over all T's
for (val in 1:nrofiters){
  allterms = 0
  j = 0
  
  #Sum from j=0 to T-1
  for (i in 0:(val-1)){
    term <- a*((1-a)^j)*dailyprevalencecopy[val-j]
    allterms <- allterms + term
    j = j+1
  }
  yvals = append(yvals, allterms)
  
  #If there is no existing datapoint, add the prediction to the list
  if (val >= length(dailycases)){
    dailyprevalencecopy = append(dailyprevalencecopy, allterms)
  }
}

#plot(x)
plot(yvals, ylim=c(-0.025, 0.20), type = "l", main="SES Prevalence using weighted average form", xlab = "days", ylab="prevalence")
summary(dailyprevalencecopy)
#print(yvals)


#Loop over a range of Days/alphas
days = c(1:100)
MSEvalues = c()
for (N in days){
  dailyprevalencecopy <- dailyprevalence #We create a copy because the forecasts of future daily prevalences are added to the daily prevalence list 
  a= 2/(N+1)
  #a = 0.09
  yvals = c(0)
  nrofiters = 557 #predict 100 days into the future
  #loop over all T's
  for (val in 1:nrofiters){
    allterms = 0
    j = 0
    
    #Sum from j=0 to T-1
    for (i in 0:(val-1)){
      term <- a*((1-a)^j)*dailyprevalencecopy[val-j]
      allterms <- allterms + term
      j = j+1
    }
    yvals = append(yvals, allterms)
    
    #If there is no existing datapoint, add the prediction to the list
    if (val >= length(dailycases)){
      dailyprevalencecopy = append(dailyprevalencecopy, allterms)
    }
  }
  TotalMSE = 0
  N=0
  for (i in 1:length(dailyprevalence)){
    MSE = (yvals[i]-dailyprevalence[i])^2
    TotalMSE = TotalMSE + MSE
    N=N+1
  }
  TotalMSE = TotalMSE/N
  #print(TotalMSE)
  
  MSEvalues <- append(MSEvalues, TotalMSE)
}

cat("Forecast value using pre-existing library:", )

plot(days, MSEvalues, type='l', main='MSE against days using WA form SES')