#Used links: https://cran.r-project.org/web/packages/smooth/vignettes/adam.html 
#and: https://openforecast.org/adam/
#and: https://rdrr.io/github/config-i1/smooth/man/adam.html

library(greybox)
library(smooth)

#import data
data <- read.csv(file="C:\\Users\\s166646\\Downloads\\prevalentie_data.csv",sep = ';')

#transform date column into compatible format for R
data$day <- as.Date(data$day, format="%Y-%m-%d")

#introduce prevalence in data
data$prevalence <- c(data$daily_cases/data$daily_tot)
summary(data$prevalence)

#plot the prevalence
plot(data$day, data$prevalence, ylim=c(-0.025, 0.20), type='l', main='Daily Prevalence', xlab='date', ylab='prevalence')

autoadam <- auto.adam(data$prevalence, model='XXX', orders=list(ar=2,i=2,ma=2,select=TRUE),distribution="dnorm", silent=FALSE, h=12)
                      

autoadam
