# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import pickle
# import os

# copied_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
# data = pd.read_csv(copied_path)
# data.columns = data.columns.str.strip()
# data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# data.head()
# cols = [0]
# data.drop(data.columns[cols], inplace=True, axis=1)

# cols = list(data.columns.values) 
# cols.pop(cols.index('City')) 
# cols.pop(cols.index('PCOS'))
# cols.pop(cols.index('PCOS_from'))
# data = data[cols+['PCOS']]

# data['PCOS'] = data['PCOS'].map(dict(Yes = 1, No = 0))

# data.head()

# x = data.drop('PCOS', axis = 1)
# y = data['PCOS']

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# model=GaussianNB()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# y_pred

# accuracy = accuracy_score(y_test, y_pred) * 100

# accuracy

# from sklearn.metrics import confusion_matrix

# results = confusion_matrix(y_test, y_pred) 
     
# print('Confusion Matrix :')
# print(results)

# import csv
# import math
# import random

# def loadDataset(data):
#   dataset = data.values.tolist()
#   for x in range(len(dataset)-1):
#     for y in range(22):
#       dataset[x][y] = int(dataset[x][y])
  
#   return dataset

# def splitDataset(dataset, splitRatio):
#   trainSize = int(len(dataset) * splitRatio)
#   trainSet = []
#   copy = dataset  
  
#   while len(trainSet) < trainSize:
#     index = random.randrange(len(copy))
#     trainSet.append(copy.pop(index))
  
#   return [trainSet, copy]

# def separateByClass(dataset):
#     separated = {}
#     for i in range(len(dataset)):
#         vector = dataset[i]
#         if vector[-1] not in separated:
#             separated[vector[-1]] = []
#         separated[vector[-1]].append(vector)
        
#     for classValue, instances in separated.items():
#         if len(instances) == 0:
#             print(f"Warning: Class {classValue} has no instances.")
            
#     return separated

# def mean(numbers):
# #   print("mean", numbers)
#   return sum(numbers)/float(len(numbers))

# def stdev(numbers):
#     if len(numbers) <= 1:  
#         return 0.0  
#     avg = mean(numbers)
#     variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
#     return math.sqrt(variance)

# def summarize(dataset):
#     summaries = []
#     if len(dataset) > 1: 
#         summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
#         del summaries[-1]  
#     return summaries

# def summarizeByClass(dataset):
#   separated = separateByClass(dataset)
#   summaries = {}

#   for classValue, instances in separated.items():
#     summaries[classValue] = summarize(instances)
#     # print("INS",isinstance)
  
#   return summaries

# def calculateProbability(x, mean, stdev):
#   exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))

#   return (1/(math.sqrt(2*math.pi)*stdev))*exponent

# def calculateClassProbabilities(summaries, inputVector):
#   probabilities = {}

#   for classValue, classSummaries in summaries.items():
#     probabilities[classValue] = 1
#     for i in range(len(classSummaries)):
#       mean, stdev = classSummaries[i]
#       x = inputVector[i]
#       probabilities[classValue] *= calculateProbability(x, mean, stdev)

#   return probabilities

# def predict(summaries, inputVector):
#   probabilities = calculateClassProbabilities(summaries, inputVector)
#   bestLabel, bestProb = None, -1

#   for classValue, probability in probabilities.items():
#     if bestLabel is None or probability > bestProb:
#       bestProb = probability
#       bestLabel = classValue

#   return bestLabel

# def getPredictions(summaries, testSet):
#   predictions = []
  
#   for i in range(len(testSet)):
#     result = predict(summaries, testSet[i])
#     predictions.append(result)

#   return predictions

# def getAccuracy(testSet, predictions):
#   correct = 0

#   for x in range(len(testSet)):
#     if testSet[x][-1] == predictions[x]:
#       correct += 1

#   return (correct/float(len(testSet)))*100.0

# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import accuracy_score 
# from sklearn.metrics import classification_report

# def main():
#   trainingSet=[] 
#   testSet=[] 
#   split = 0.67  

#   dataset = loadDataset(data)

#   trainingSet, testSet = splitDataset(dataset, split)

#   print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))

#   summaries = summarizeByClass(trainingSet)

#   predictions = getPredictions(summaries, testSet)
#   accuracy = getAccuracy(testSet, predictions)
#   print('Accuracy: {0}%'.format(accuracy))

#   predictions = getPredictions(summaries, testSet)
#   accuracy = getAccuracy(testSet, predictions)
#   print('Accuracy: {0}%'.format(accuracy))

#   actual = []

#   for x in range(len(testSet)): 
#     actual.append(testSet[x][-1])

#   results = confusion_matrix(actual, predictions) 
  
#   print('Confusion Matrix :')
#   print(results) 
#   print('Accuracy Score :',accuracy_score(actual, predictions) )
#   print('Report : ')
#   print(classification_report(actual, predictions) )
 
# main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os
import math
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load and clean the dataset
copied_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(copied_path)
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Drop unnecessary columns
cols = [0]
data.drop(data.columns[cols], inplace=True, axis=1)
cols = list(data.columns.values)
cols.pop(cols.index('City'))
cols.pop(cols.index('PCOS'))
cols.pop(cols.index('PCOS_from'))
data = data[cols + ['PCOS']]

# Encode 'PCOS' labels
data['PCOS'] = data['PCOS'].map(dict(Yes=1, No=0))

# Split data into X (features) and y (target)
x = data.drop('PCOS', axis=1)
y = data['PCOS']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# ** 1. Sklearn Gaussian Naive Bayes model (saving after training) **
model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Calculate accuracy for sklearn model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Sklearn GaussianNB Accuracy: {accuracy}%")

# Save the sklearn Naive Bayes model
directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
os.makedirs(directory, exist_ok=True)

filename = f'{directory}/NB_model.pkl'
pickle.dump(model, open(filename, 'wb'))
# with open('naive_bayes_model.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)
#     print("Sklearn GaussianNB model saved successfully.")

# ** 2. Custom Naive Bayes Implementation (saving learned summaries) **

def loadDataset(data):
    dataset = data.values.tolist()
    for x in range(len(dataset)-1):
        for y in range(22):  # Assuming 22 features in the dataset
            dataset[x][y] = int(dataset[x][y])
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = dataset  
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    if len(numbers) <= 1:
        return 0.0  
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = []
    if len(dataset) > 1: 
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]  # Removing the last column (the class label)
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# Load dataset and prepare for custom Naive Bayes
dataset = loadDataset(data)
trainingSet, testSet = splitDataset(dataset, 0.67)
summaries = summarizeByClass(trainingSet)

# Make predictions and evaluate accuracy for custom Naive Bayes
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print(f"Custom Naive Bayes Accuracy: {accuracy}%")

# Save the custom Naive Bayes model (summaries)
directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
os.makedirs(directory, exist_ok=True)

filename = f'{directory}/Custom_NB_model.pkl'
pickle.dump(summaries, open(filename, 'wb'))
feature_names_path = f'{directory}/Custom_NB_features.pkl'
pickle.dump(cols, open(feature_names_path, 'wb'))  

# Evaluation of confusion matrix and classification report
actual = [testSet[i][-1] for i in range(len(testSet))]
results = confusion_matrix(actual, predictions)
print('Confusion Matrix:')
print(results)
print('Classification Report:')
print(classification_report(actual, predictions))
