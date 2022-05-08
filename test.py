import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#rawdata = pd.read_csv(r'C:\Users\s166646\Downloads\seh_data.csv', sep = ';')
#data = pd.DataFrame(rawdata)

#data['prevalence'] = data['daily_cases']/data['daily_tot']
#print(data)

#plt.plot(data['prevalence'])
#plt.show()


def importData(path):
    '''Function that imports the data from a given filepath and implements daily prevalence
    Input: string of data file path 
    Output: dataframe with clinical data
    '''
    
    #import data
    rawdata = pd.read_csv(path, sep=';')
    data = pd.DataFrame(rawdata)


    #Introduce prevalence
    data['prevalence'] = data['daily_cases']/data['daily_tot']

    return data

def splitData(data, trainingFrac):
    '''Function that splits the data into two fractions
    Input: data to be split, desired fraction to split the data 
    Output: two dataframes
    '''

    trainDataInd = round(trainingFrac * len(data))
    testDataInd = len(data) - trainDataInd

    traindata = pd.DataFrame(data[0:trainDataInd])
    testdata = pd.DataFrame(data[-testDataInd:])

    #print(traindata)

    return traindata, testdata

def getSESpredictions(alpha, variable):
    '''Function that fits the SES model on a variable
    Input: alpha parameter for the model, the variable to fit the model on
    Output: The fitted values of the SES model
    '''
    lArray = [0]

    for i in range(len(variable)-1):
        et = variable[i] - lArray[i-1]
        lArrayVal = lArray[i-1] + alpha*et
        lArray.append(lArrayVal)

    return lArray[1:]

    #Put the starting value on 0
    #predictedVals = np.array([0])
    
    #Loop over the data, the range statement is like this so that it can work on differently indexed folds in the data e.g. index 40-80 instead of index 0-x
    #for count, i in enumerate(range((variable.index[-1]-len(variable)+1), variable.index[-1]+1)):
    #    et = variable[i] - predictedVals[count-1]
    #    predictedVal = predictedVals[count-1] + alpha*et
    #    predictedVals = np.append(predictedVals, predictedVal)
    #predictedVals = np.append(0, predictedVals)

    #predictedVals = predictedVals[1:]

    #return predictedVals

def getMSE(vectorA, vectorB):
    '''Function that calculates the MSE between two vectors. If one is longer than the other, the MSE will be calculated over the equal part.
    Input: two vectors, don't have to be equal length
    Output: MSE
    '''

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

def flatten(listOfLists):
    '''Function that flattens lists of lists to a single list. For example [[1,2],[3,4],[5,6]] to [1,2,3,4,5,6]
    '''
    flatList = []
    for list in listOfLists:
        for item in list:
            flatList.append(item)
    return flatList


#import the data
data = importData(r'C:\Users\s166646\Downloads\seh_data.csv')

#Split into training/validation and testing datasets
trainingDataSet, testDataSet = splitData(data, 0.8)

#Plot the data
#plt.plot(trainingDataSet['prevalence'])
#plt.show()

#Perform crossvalidation

#Set the desired number of folds
nrOfFolds = 10

#Split the data into folds
dataInds = list(range(len(trainingDataSet)))
foldInds = np.array_split(dataInds, nrOfFolds)

#Determine the grid of alpha's to be tried on the training set
alphaGrid = [x/100 for x in range(1, 100, 1)]

#Initialize lists with the MSEs on the validation sets and the alpha that was used
listOfValidationMSEs = []
validationAlphas = []

#Loop over the folds
for i in range(nrOfFolds-1):

    #Initialize list of the MSEs for all the alpha's in the grid
    listOfMSEs = []

    #Get the training set for this particular fold
    unflattenedTrainingInds = foldInds[:i+1]
    flattenedTrainingInds = flatten(unflattenedTrainingInds)
    trainingData = trainingDataSet.iloc[flattenedTrainingInds]

    #Get the validation set for this particular fold
    unflattenedValidationInds = foldInds[i+1:i+2]
    flattenedValidationInds = flatten(unflattenedValidationInds)
    validationData = trainingDataSet.iloc[flattenedValidationInds]

    #Reindex the validation set
    labels = list(range(0,len(validationData)))
    validationData.set_index([pd.Index(labels)], inplace=True, drop=True)
    
    #Try all the alphas on the training set of this fold and store their MSEs
    for alpha in alphaGrid:
        trainingPredictions = getSESpredictions(alpha, trainingData['prevalence'])
        foldMSE = getMSE(trainingPredictions, trainingData['prevalence'])

        listOfMSEs.append(foldMSE)
    
    #Get the best alpha and corresponding MSE
    foldsBestMSE = min(listOfMSEs)
    foldsBestAlphaIndex = listOfMSEs.index(foldsBestMSE)
    foldsBestAlpha = alphaGrid[foldsBestAlphaIndex]
    validationAlphas.append(foldsBestAlpha)

    print('The alpha with the lowest MSE on the training set:', foldsBestAlpha)
    print('The lowest MSE on the training set:', foldsBestMSE)

    #For my own research, fit the best training set alpha/MSE to see how it looks 
    bestTrainingPredictions = getSESpredictions(foldsBestAlpha, trainingData['prevalence'])
    bestTrainingPredictions.insert(0,0)
    #Compare with pre-made package SES
    trainingTest = SimpleExpSmoothing(trainingData['prevalence'], initialization_method="known", initial_level=0).fit(smoothing_level=foldsBestAlpha, optimized=False)

    print(len(trainingData['prevalence']))
    #Plot training set performance
    #plt.plot(trainingData['prevalence'], 'g-', label='actual values')
    plt.plot(bestTrainingPredictions, 'b-', label = 'self-made SES')
    plt.plot(trainingTest.fittedvalues, 'r-', label='premade package SES')
    plt.title('SES on the training set of fold {} '.format(i))
    plt.xlabel('Day')
    plt.ylabel('Prevalence')
    plt.legend(loc='upper left')
    plt.show()


    #Use alpha that performed best on the training set and fit it on the validation set
    validationPredictions = getSESpredictions(foldsBestAlpha, validationData['prevalence'])
    validationTest = SimpleExpSmoothing(validationData['prevalence'], initialization_method="known", initial_level=0).fit(smoothing_level=foldsBestAlpha, optimized=False)
    
    #print(validationPredictions)
    #print(validationData['prevalence'])
    validationMSE = getMSE(validationPredictions, validationData['prevalence'])
    listOfValidationMSEs.append(validationMSE)

    print(len(validationPredictions))
    print(len(validationData['prevalence']))

    plt.plot(bestTrainingPredictions)
    plt.plot(trainingData['prevalence'])
    plt.plot(trainingTest.fittedvalues)
    plt.show()

    plt.plot(validationData['prevalence'])
    plt.plot(validationPredictions)
    plt.plot(validationTest.fittedvalues)
    plt.show()

plt.plot((validationAlphas,listOfValidationMSEs))
plt.show()





