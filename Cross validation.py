import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from statsmodels.tsa.ar_model import ar_select_order
import statsmodels.api as sm

class Crossvalidation:
    def __init__(self, rIVMLocation, cZELocation):
        #Get the files
        self.rIVMLocation = rIVMLocation
        self.cZELocation = cZELocation

        #Convert RIVM to dataframe
        rawRIVMData = pd.read_csv(self.rIVMLocation, sep=';')
        self.trainingSet = pd.DataFrame(rawRIVMData)

        #Convert CZE to dataframe
        rawCZEData = pd.read_csv(self.cZELocation, sep=';')
        self.testSet = pd.DataFrame(rawCZEData)

        #Introduce prevalence in CZE data
        self.testSet['prevalence'] = self.testSet['daily_cases']/self.testSet['daily_tot']

    def datasmoother(self, data, days):
        data= list(data)
        returnlist = []
        list1 = list2 = []
        for n in range(days):
            list1.append(data[0])
            list2.append(data[len(data)-1])
        newdata = list1+data+list2
        for i in range(days+3,len(data)+6):
            val = newdata[i]
            for n in range(days):
                n +=1
                newvals = newdata[i+n] + newdata[i-n]
                val +=newvals
            val = val/(days*2)
            returnlist.append(val)
        return returnlist

    def getMSE(self, vectorA, vectorB):
        mse = 0

        #Statements to prevent errors when the lengts are unequal
        if len(vectorA) <= len(vectorB):
            minLen = len(vectorA)
        else:
            minLen = len(vectorB)

        for i in range(minLen):
            diff = (vectorA[i] - vectorB[i])**2
            mse += diff

        mse = mse / minLen
        return mse

    def getSESpredictions(self, data, alpha):

        predictions = SimpleExpSmoothing(data, initialization_method="known", initial_level=0).fit(smoothing_level=alpha, optimized=False)
        returnvalues = predictions.fittedvalues
        
        return returnvalues

    def getARIMApredictions(self, data, alpha):

        predictions = ARIMA(data, order=alpha, enforce_stationarity=False)
        model = predictions.fit()

        return model.fittedvalues

    def getKalmanPredictions(self, data, alpha):

        #Initialization
        q = alpha[0]
        me= alpha[1] 

        xold = 0.01
        pold = 0.075

        predictions = [xold]
        uncertainties = [pold]

        for i in range(len(data)):
            zn = data[i]
            kGain = pold/(pold+me)
            xnew = xold + kGain*(zn-xold)
            pcurrent = (1-kGain)*pold

            predictions.append(xnew)
            pnew = pcurrent+q
            uncertainties.append(pnew)
            pold = pnew
            xold = xnew
    
        return predictions

    def getSETARpredictions(self, data, alpha):
        finald = alpha[0]
        finalgamma = alpha[1]*max(data)

        lowvalues = []
        highvalues = []

        regimesperind = []

        for i in range(len(data)):
            if i-finald<=0:
                regimesperind.append(0)
                lowvalues.append(data[i])
            if i-finald > 0:
                if statistics.mean(data[i-finald:i]) <= finalgamma:
                    regimesperind.append(0)
                    lowvalues.append(data[i])
                if statistics.mean(data[i-finald:i]) > finalgamma:
                    regimesperind.append(1)
                    highvalues.append(data[i])

        lowmodelorder = ar_select_order(lowvalues, maxlag=10, ic='aic')
        highmodelorder = ar_select_order(highvalues, maxlag=10, ic='aic')

        if 1 not in lowmodelorder.ar_lags:
            loworder = 1
        else:
            loworder = max(lowmodelorder.ar_lags)
        
        if 1 not in highmodelorder.ar_lags:
            highorder = 1
        else:
            highorder = max(highmodelorder.ar_lags)

        #Initialize AR models 
        lowmodel = ARIMA(lowvalues, order=(loworder,0,0), enforce_stationarity=False)
        highmodel = ARIMA(highvalues, order=(highorder,0,0), enforce_stationarity=False)

        #Fit the models
        lowresults = lowmodel.fit(method='innovations_mle')
        highresults = highmodel.fit(method='innovations_mle')

        lowpredictions = lowresults.fittedvalues
        highpredictions = highresults.fittedvalues

        finalpredictions = []

        for i in range(len(regimesperind)):

            if regimesperind[i] == 0:
                finalpredictions.append(lowpredictions[0])
                lowpredictions = lowpredictions[1:]
            
            if regimesperind[i] == 1:
                finalpredictions.append(highpredictions[0])
                highpredictions = highpredictions[1:] 
    
        return finalpredictions

    def getARIMASESpredictions(self, data, alpha):
        sesmodel = SimpleExpSmoothing(data, initialization_method="known", initial_level=0).fit()
        sespredictions = sesmodel.fittedvalues

        arimamodel = ARIMA(data, order=alpha, enforce_stationarity=False)
        arimamodelfit = arimamodel.fit()
        arimapredictions = arimamodelfit.fittedvalues

        combinedpredictions = []

        for i in range(len(arimapredictions)):
            combinedpredictions.append((arimapredictions[i]+sespredictions[i])/2)

        return combinedpredictions

    def getMarkovARIMApredictions(self, data, alpha):
        model = sm.tsa.MarkovRegression(data, k_regimes=2)
        results = model.fit(em_iter=10)
    
        regimes = results.smoothed_marginal_probabilities[1]
        regimes = np.round(regimes)

        lowmodel = ARIMA(data, order=alpha[0], enforce_stationarity=False)
        highmodel = ARIMA(data, order=alpha[1], enforce_stationarity=False)

        lowfit = lowmodel.fit()
        highfit = highmodel.fit()

        lowpreds = np.array(lowfit.fittedvalues)
        highpreds = np.array(highfit.fittedvalues)

        finalpreds = []

        for i,nr in enumerate(regimes):
            if nr == 0:
                finalpreds.append(lowpreds[i])
            if nr ==1:
                finalpreds.append(highpreds[i])

        return finalpreds

    def crossvalidate(self, nrOfFolds, smoothingparameter, model, modelparameters, showTrainingPredictions = False):
        #Create the "golden standard" values 
        testSetSmoothedPrevalence = self.datasmoother(data= self.testSet['prevalence'], days= smoothingparameter)
        trainingSetSmoothed = pd.DataFrame()

        for column in self.trainingSet:
            trainingSetSmoothed[column] = self.datasmoother(self.trainingSet[column],smoothingparameter)
        
        #Initialize lists we're gonna need for analysis
        bestParameters = []
        bestMSEs = []

        validationMSEs = []
        validationParameters = []

        #Start at fold 0
        valFoldNr = 0
        #Perform crossvalidation
        while valFoldNr < nrOfFolds:
            #Initialize indexes for training/validation splits
            dataInds = list(range(25))
            foldInds = np.array_split(dataInds, nrOfFolds)

            #Create validation folds
            validationFold = foldInds[valFoldNr]
            trainingFolds = foldInds
            pop = trainingFolds.pop(valFoldNr)
    
            trainingSeriesInds = []
            validationSeriesInds = []

            #Get lists with specifically the indexes of training and validation samples
            for fold in trainingFolds:
                for i in fold:
                    trainingSeriesInds.append(i)
                    #Train zo dat de uiteindelijke alpha de beste is over alle 5 de train sets

            for i in validationFold:
                validationSeriesInds.append(i)

            #Initialize the alphas we wanna train the model with
            alphas= modelparameters#[x/100 for x in range(1, 100, 1)]

            totalMSEs = np.zeros(len(alphas))
            trainingPerformances = []

            for i in trainingSeriesInds:
                alphaMSEs = []

                #Try all alphas for one series, store the MSE of each alpha in the list 'alphaMSEs'
                for alpha in alphas:
                    #print(alpha)
                    if model=='SES':
                        predictions = self.getSESpredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)
                        mse = self.getMSE(trainingSetSmoothed.iloc[:,i], predictions)
                        #predictions = SimpleExpSmoothing(self.trainingSet.iloc[:,i], initialization_method="known", initial_level=0).fit(smoothing_level=alpha, optimized=False)
                        #mse = self.getMSE(trainingSetSmoothed.iloc[:,i],predictions.fittedvalues)

                        #plt.plot(predictions, label='predictions')
                        #plt.plot(trainingSetSmoothed.iloc[:,i])
                        #plt.ylabel('prevalence')
                        #plt.xlabel('alpha: {}'.format(alpha))
                        #plt.show()

                    if model=='ARIMA':
                        #print('alpha:',alpha)
                        predictions = self.getARIMApredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)
                        mse = self.getMSE(trainingSetSmoothed.iloc[:,i], predictions)

                        #plt.plot(predictions, label='predictions')
                        #plt.plot(trainingSetSmoothed.iloc[:,i])
                        #plt.ylabel('prevalence')
                        #plt.xlabel('alpha: {}'.format(alpha))
                        #plt.show()

                    if model == 'Kalman':
                        predictions = self.getKalmanPredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)

                    if model == 'SETAR':
                        predictions = self.getSETARpredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)

                    if model == 'ARIMASES':
                        predictions = self.getARIMASESpredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)

                    if model == 'MarkovARIMA':
                        predictions = self.getMarkovARIMApredictions(data=self.trainingSet.iloc[:,i], alpha=alpha)

                    mse = self.getMSE(trainingSetSmoothed.iloc[:,i], predictions)
                    if showTrainingPredictions == True:
                        plt.plot(predictions, label='predictions')
                        plt.plot(trainingSetSmoothed.iloc[:,i], label='Golden standard, smoothed over: {} days'.format(smoothingparameter*2+1))
                        plt.ylabel('prevalence')
                        plt.xlabel('alpha: {}'.format(alpha))
                        plt.legend()
                        plt.show()

                    alphaMSEs.append(mse)

                trainingPerformances.append(alphaMSEs)
                totalMSEs = [x + y for (x, y) in zip(alphaMSEs, totalMSEs)]

            #Pick the best alpha and MSE out of the training set
            bestMSE = min(totalMSEs)
            bestAlphaIndex = totalMSEs.index(bestMSE)
            bestAlpha = alphas[bestAlphaIndex]

            print('Best training MSE:', bestMSE)
            print('Best training Alpha:', bestAlpha)

            validationSetMSEs = 0
            #validationSetMSEs = []
                
            for i in validationSeriesInds:
                if model == 'SES':

                    #predictions = getSESpredictions(bestAlpha, trainingSet.iloc[:,i])
                    predictions = self.getSESpredictions(self.trainingSet.iloc[:,i], bestAlpha)
                    mse = self.getMSE(predictions, trainingSetSmoothed.iloc[:,i])
                    #mse = self.getMSE(predictions, trainingSetSmoothed.iloc[:,i])
                    #validationMSEs.append(mse)
                    #validationParameters.append(bestAlpha)

                if model == 'ARIMA':
                    predictions = self.getARIMApredictions(data=self.trainingSet.iloc[:,i], alpha=bestAlpha)

                if model == 'Kalman':
                    predictions = self.getKalmanPredictions(self.trainingSet.iloc[:,i], bestAlpha)

                if model == 'SETAR':
                    predictions = self.getSETARpredictions(self.trainingSet.iloc[:,i], bestAlpha)

                if model == 'ARIMASES':
                    predictions = self.getARIMASESpredictions(data=self.trainingSet.iloc[:,i], alpha=bestAlpha)
                    
                if model == 'MarkovARIMA':
                    predictions = self.getMarkovARIMApredictions(data=self.trainingSet.iloc[:,i], alpha=bestAlpha)

                mse = self.getMSE(predictions, trainingSetSmoothed.iloc[:,i])
                validationMSEs.append(mse)
                validationParameters.append(bestAlpha)
                validationSetMSEs += mse

            validationSetMSEs = validationSetMSEs/len(validationSeriesInds)

            print('Mean MSE on validation set:', validationSetMSEs)
            print('alpha used:', bestAlpha)    

            #validationMSEs.append(validationSetMSEs)
            #validationMSEs.append(mse)
            #validationParameters.append(bestAlpha)

            valFoldNr += 1

        #plt.scatter(validationParameters,validationMSEs)
        
        plt.scatter(range(len(validationMSEs)),validationMSEs)
        plt.ylabel('MSE')
        plt.xlabel('Dataset number')
        plt.title('MSEs on validation sets')
        plt.show()

        print('mean validation set MSE:', statistics.mean(validationMSEs))
        #print('Std err on validation set MSE:', statistics.stdev(validationMSEs))
        print('Std err on validation set MSE:', stats.sem(validationMSEs))
        print('alphas used:', validationParameters)

        finalAlpha = statistics.median(validationParameters)

        if model=='SES':
            testPredictions = self.getSESpredictions(self.testSet['prevalence'],finalAlpha)

        if model=='ARIMA':
            testPredictions = self.getARIMApredictions(data=self.testSet['prevalence'], alpha=finalAlpha)

        if model=='Kalman':
            testPredictions = self.getKalmanPredictions(data=self.testSet['prevalence'], alpha=finalAlpha)

        if model=='SETAR':
            testPredictions = self.getSETARpredictions(data=self.testSet['prevalence'], alpha=finalAlpha)

        if model=='ARIMASES':
            testPredictions = self.getARIMASESpredictions(data=self.testSet['prevalence'], alpha=finalAlpha)

        if model == 'MarkovARIMA':
            testPredictions = self.getMarkovARIMApredictions(data=self.testSet['prevalence'], alpha=finalAlpha)

        testSetMSE = self.getMSE(testPredictions, testSetSmoothedPrevalence)
        print('final alpha used:', finalAlpha)
        print('test set MSE:', testSetMSE)

        #plt.plot(testSet['prevalence'], label='True values')
        plt.plot(testPredictions, label='predictions')
        plt.plot(testSetSmoothedPrevalence, label='Golden standard values')
        plt.ylabel('prevalence')
        plt.xlabel('day')
        plt.title('Predictions on the CZE set, smoothed over {} days'.format(smoothingparameter*2+1))
        plt.legend()
        plt.show()

        #plt.plot(self.testSet['prevalence'], label= 'Actual values')
        #plt.plot(testSetSmoothedPrevalence, label= 'Golden standard values')
        #plt.title('CZE set, prevalence smoothed over {} days'.format(smoothingparameter*2+1))
        #plt.ylabel('prevalence')
        #plt.xlabel('day')
        #plt.legend()
        #plt.show()

#Model parameters Kalman
qs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
mes = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
modelparametersKalman = []

for q in qs:
    for m in mes:
        modelparametersKalman.append([q,m])

modelparametersSETAR = []

ds = [1,2,3,4,5,6]
gammas = [0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25]
for d in ds:
    for gamma in gammas:
        modelparametersSETAR.append([d,gamma])

#Model parameters combined ses/arima
modelparametersCombined = []
#sesparameters = [x/100 for x in range(1, 100, 1)]
sesparameters = [0.05, 0.1, 0.15, 0.2,  0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
arimaparameters = [(0, 0, 1) ,(0, 0, 2) ,(0, 0, 3)  ,(0, 1, 1) ,(0, 1, 2) ,(0, 1, 3) ,(0, 2, 1) ,(0, 2, 2) ,(0, 2, 3) ,(1, 0, 0) ,(1, 0, 1) ,(1, 0, 2) ,(1, 0, 3) ,(1, 1, 0) ,(1, 1, 1) ,(1, 1, 2) ,(1, 1, 3) ,(1, 2, 0) ,(1, 2, 1) ,(1, 2, 2) ,(1, 2, 3) ,(2, 0, 0) ,(2, 0, 1) ,(2, 0, 2) ,(2, 0, 3) ,(2, 1, 0) ,(2, 1, 1) ,(2, 1, 2) ,(2, 1, 3) ,(2, 2, 0) ,(2, 2, 1) ,(2, 2, 2) ,(2, 2, 3) ,(3, 0, 0) ,(3, 0, 1) ,(3, 0, 2) ,(3, 0, 3) ,(3, 1, 0) ,(3, 1, 1) ,(3, 1, 2) ,(3, 1, 3) ,(3, 2, 0) ,(3, 2, 1) ,(3, 2, 2) ,(3, 2, 3)]

for sesparam in sesparameters:
    for arimaparam in arimaparameters:
        modelparametersCombined.append([sesparam,arimaparam])

#Markov-ARIMA parameters
markovparameters = []

for a in arimaparameters:
    for b in arimaparameters:
        markovparameters.append([a,b])

exp1 = Crossvalidation(r'C:\Users\s166646\Downloads\Gemeentes.csv', r'C:\Users\s166646\Downloads\seh_data.csv')

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='SES', modelparameters=[x/100 for x in range(1, 100, 1)])

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='SES', modelparameters=[x/1000 for x in range(250, 460, 1)])
#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='SES', modelparameters=[x/1000 for x in range(210, 360, 1)])


#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='ARIMA', modelparameters=[(0, 0, 1) ,(0, 0, 2) ,(0, 0, 3)  ,(0, 1, 1) ,(0, 1, 2) ,(0, 1, 3) ,(0, 2, 1) ,(0, 2, 2) ,(0, 2, 3) ,(1, 0, 0) ,(1, 0, 1) ,(1, 0, 2) ,(1, 0, 3) ,(1, 1, 0) ,(1, 1, 1) ,(1, 1, 2) ,(1, 1, 3) ,(1, 2, 0) ,(1, 2, 1) ,(1, 2, 2) ,(1, 2, 3) ,(2, 0, 0) ,(2, 0, 1) ,(2, 0, 2) ,(2, 0, 3) ,(2, 1, 0) ,(2, 1, 1) ,(2, 1, 2) ,(2, 1, 3) ,(2, 2, 0) ,(2, 2, 1) ,(2, 2, 2) ,(2, 2, 3) ,(3, 0, 0) ,(3, 0, 1) ,(3, 0, 2) ,(3, 0, 3) ,(3, 1, 0) ,(3, 1, 1) ,(3, 1, 2) ,(3, 1, 3) ,(3, 2, 0) ,(3, 2, 1) ,(3, 2, 2) ,(3, 2, 3)])
exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='ARIMA', modelparameters=[(0, 0, 1) ,(0, 0, 2) ,(0, 0, 3)  ,(0, 1, 1) ,(0, 1, 2) ,(0, 1, 3) ,(0, 2, 1) ,(0, 2, 2) ,(0, 2, 3) ,(1, 0, 0) ,(1, 0, 1) ,(1, 0, 2) ,(1, 0, 3) ,(1, 1, 0) ,(1, 1, 1) ,(1, 1, 2) ,(1, 1, 3) ,(1, 2, 0) ,(1, 2, 1) ,(1, 2, 2) ,(1, 2, 3) ,(2, 0, 0) ,(2, 0, 1) ,(2, 0, 2) ,(2, 0, 3) ,(2, 1, 0) ,(2, 1, 1) ,(2, 1, 2) ,(2, 1, 3) ,(2, 2, 0) ,(2, 2, 1) ,(2, 2, 2) ,(2, 2, 3) ,(3, 0, 0) ,(3, 0, 1) ,(3, 0, 2) ,(3, 0, 3) ,(3, 1, 0) ,(3, 1, 1) ,(3, 1, 2) ,(3, 1, 3) ,(3, 2, 0) ,(3, 2, 1) ,(3, 2, 2) ,(3, 2, 3)])

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='Kalman', modelparameters=modelparametersKalman, showTrainingPredictions=False)
#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='Kalman', modelparameters=modelparametersKalman, showTrainingPredictions=False)

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='SETAR', modelparameters=modelparametersSETAR, showTrainingPredictions=False)

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='ARIMASES', modelparameters=arimaparameters, showTrainingPredictions=False)
#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='ARIMASES', modelparameters=arimaparameters, showTrainingPredictions=False)

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='MarkovARIMA', modelparameters=markovparameters, showTrainingPredictions=True)