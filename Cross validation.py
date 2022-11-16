import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

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
                    mse = self.getMSE(predictions, trainingSetSmoothed.iloc[:,i])

                if model == 'Kalman':
                    predictions = self.getKalmanPredictions(self.trainingSet.iloc[:,i], bestAlpha)
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
        plt.show()

        print('mean validation set MSE:', statistics.mean(validationMSEs))
        #print('Std err on validation set MSE:', statistics.stdev(validationMSEs))
        print('Std err on validation set MSE:', stats.sem(validationMSEs))
        print('alphas used:', validationParameters)

        finalAlpha = statistics.median(validationParameters)

        if model=='SES':
            testPredictions = self.getSESpredictions(self.testSet['prevalence'],finalAlpha)
            testSetMSE = self.getMSE(testPredictions, testSetSmoothedPrevalence)

        if model=='ARIMA':
            testPredictions = self.getARIMApredictions(data=self.testSet['prevalence'], alpha=finalAlpha)
            testSetMSE = self.getMSE(testPredictions, testSetSmoothedPrevalence)

        if model=='Kalman':
            testPredictions = self.getKalmanPredictions(data=self.testSet['prevalence'], alpha=finalAlpha)
            testSetMSE = self.getMSE(testPredictions, testSetSmoothedPrevalence)

        print('final alpha used:', finalAlpha)
        print('test set MSE:', testSetMSE)

        #plt.plot(testSet['prevalence'], label='True values')
        plt.plot(testPredictions, label='predictions')
        plt.plot(testSetSmoothedPrevalence, label='Golden standard values')
        plt.ylabel('prevalence')
        plt.xlabel('day')
        plt.legend()
        plt.show()


#Model parameters Kalman
qs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
mes = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
modelparametersKalman = []

for q in qs:
    for m in mes:
        modelparametersKalman.append([q,m])



exp1 = Crossvalidation(r'C:\Users\s166646\Downloads\Gemeentes.csv', r'C:\Users\s166646\Downloads\seh_data.csv')

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='SES', modelparameters=[x/100 for x in range(1, 100, 1)])

#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='SES', modelparameters=[x/1000 for x in range(250, 360, 1)])
#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='SES', modelparameters=[x/1000 for x in range(210, 360, 1)])


#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='ARIMA', modelparameters=[(0, 0, 1) ,(0, 0, 2) ,(0, 0, 3)  ,(0, 1, 1) ,(0, 1, 2) ,(0, 1, 3) ,(0, 2, 1) ,(0, 2, 2) ,(0, 2, 3) ,(1, 0, 0) ,(1, 0, 1) ,(1, 0, 2) ,(1, 0, 3) ,(1, 1, 0) ,(1, 1, 1) ,(1, 1, 2) ,(1, 1, 3) ,(1, 2, 0) ,(1, 2, 1) ,(1, 2, 2) ,(1, 2, 3) ,(2, 0, 0) ,(2, 0, 1) ,(2, 0, 2) ,(2, 0, 3) ,(2, 1, 0) ,(2, 1, 1) ,(2, 1, 2) ,(2, 1, 3) ,(2, 2, 0) ,(2, 2, 1) ,(2, 2, 2) ,(2, 2, 3) ,(3, 0, 0) ,(3, 0, 1) ,(3, 0, 2) ,(3, 0, 3) ,(3, 1, 0) ,(3, 1, 1) ,(3, 1, 2) ,(3, 1, 3) ,(3, 2, 0) ,(3, 2, 1) ,(3, 2, 2) ,(3, 2, 3)])
#exp1.crossvalidate(nrOfFolds=5, smoothingparameter=6, model='ARIMA', modelparameters=[(0, 0, 1) ,(0, 0, 2) ,(0, 0, 3)  ,(0, 1, 1) ,(0, 1, 2) ,(0, 1, 3) ,(0, 2, 1) ,(0, 2, 2) ,(0, 2, 3) ,(1, 0, 0) ,(1, 0, 1) ,(1, 0, 2) ,(1, 0, 3) ,(1, 1, 0) ,(1, 1, 1) ,(1, 1, 2) ,(1, 1, 3) ,(1, 2, 0) ,(1, 2, 1) ,(1, 2, 2) ,(1, 2, 3) ,(2, 0, 0) ,(2, 0, 1) ,(2, 0, 2) ,(2, 0, 3) ,(2, 1, 0) ,(2, 1, 1) ,(2, 1, 2) ,(2, 1, 3) ,(2, 2, 0) ,(2, 2, 1) ,(2, 2, 2) ,(2, 2, 3) ,(3, 0, 0) ,(3, 0, 1) ,(3, 0, 2) ,(3, 0, 3) ,(3, 1, 0) ,(3, 1, 1) ,(3, 1, 2) ,(3, 1, 3) ,(3, 2, 0) ,(3, 2, 1) ,(3, 2, 2) ,(3, 2, 3)])

exp1.crossvalidate(nrOfFolds=5, smoothingparameter=3, model='Kalman', modelparameters=modelparametersKalman, showTrainingPredictions=False)