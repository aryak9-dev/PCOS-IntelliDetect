# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# import operator
# import math
# import random
# import csv
# import pandas as pd
# import numpy as np
# import os
# import pickle  # To save the model

# copied_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
# data = pd.read_csv(copied_path)

# data.head()
# cols = [0]
# data.drop(data.columns[cols], inplace=True, axis=1)

# data.columns = data.columns.str.strip()
# data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# cols = list(data.columns.values) 
# cols.pop(cols.index('City'))  
# cols.pop(cols.index('PCOS'))
# cols.pop(cols.index('PCOS_from'))
# data = data[cols+['City', 'PCOS_from', 'PCOS']]

# data['PCOS'] = data['PCOS'].map(dict(Yes=1, No=0))

# data.head()

# def loadDataset(data, split, trainingSet=[], testSet=[]):
#     dataset = data.values.tolist()

#     label_encoder = LabelEncoder()
    
#     categorical_columns = ['City']  

#     for cat_col in categorical_columns:
#         data[cat_col] = label_encoder.fit_transform(data[cat_col])
    
#     for x in range(len(dataset) - 1):
#         for y in range(len(dataset[x])):
            
#             if isinstance(dataset[x][y], str) and dataset[x][y].strip() == '':
#                 dataset[x][y] = 0  
#             try:
#                 dataset[x][y] = int(dataset[x][y]) 
#             except ValueError:
#                 continue  

#         if random.random() < split:
#             trainingSet.append(dataset[x])
#         else:
#             testSet.append(dataset[x])

# def euclideanDistance(instance1, instance2, length):
#     distance = 0
#     for x in range(length):
#         distance += pow((instance1[x] - instance2[x]), 2)
#         return math.sqrt(distance)


# def getNeighbors(trainingSet, testInstance, k):
#     distances = []
#     length = len(testInstance)-1
#     for x in range(len(trainingSet)):
#         dist = euclideanDistance(testInstance, trainingSet[x], length)
#         distances.append((trainingSet[x], dist))
#         distances.sort(key=operator.itemgetter(1))
#         neighbors = []

#     for x in range(k):
#         neighbors.append(distances[x][0])

#     return neighbors

# def getResponse(neighbors):
#     classVotes = {}
#     for x in range(len(neighbors)):
#         response = neighbors[x][-1]

#         if response in classVotes:
#             classVotes[response] += 1
#         else:
#             classVotes[response] = 1
#             sortedVotes = sorted(classVotes.items(),
#                                  key=operator.itemgetter(1), reverse=True)

#     return sortedVotes[0][0]


# def getAccuracy(testSet, predictions):
#     correct = 0

#     for x in range(len(testSet)):
#         if testSet[x][-1] == predictions[x]:
#             correct += 1

#     return (correct/float(len(testSet)))

# def main():  
#     trainingSet = []
#     testSet = []
#     split = 0.67

#     loadDataset(data, split, trainingSet, testSet)

#     print('Train set: ' + repr(len(trainingSet)))

#     print('Test set: ' + repr(len(testSet)))

#     predictions = []
#     k = 3

#     for x in range(len(trainingSet)):
#         neighbors = getNeighbors(trainingSet, trainingSet[x], k)
#         result = getResponse(neighbors)
#         predictions.append(result)

#     accuracy = getAccuracy(trainingSet, predictions)
#     print('Accuracy Training: ' + repr(accuracy) + '%')

#     predictions.clear()
#     for x in range(len(testSet)):
#         neighbors = getNeighbors(trainingSet, testSet[x], k)
#         result = getResponse(neighbors)
#         predictions.append(result)

#     accuracy = getAccuracy(testSet, predictions)
#     print('Accuracy Test: ' + repr(accuracy) + '%')

#     actual = []

#     for x in range(len(testSet)):
#         actual.append(testSet[x][-1])

#     results = confusion_matrix(actual, predictions)

#     print('Confusion Matrix :')
#     print(results)
#     print('Accuracy Score :', accuracy_score(actual, predictions))
#     print('Report : ')
#     print(classification_report(actual, predictions))

#        # ** Saving the KNN Model **
#     model_data = {
#         "trainingSet": trainingSet,
#         "k": k
#     }

#     directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
#     os.makedirs(directory, exist_ok=True)

#     filename = f'{directory}/knn_model.pkl'
#     pickle.dump(model_data, open(filename, 'wb'))




# main()


# impfrom sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import operator
import math
import random
import csv
import pandas as pd
import numpy as np
import os
import pickle

copied_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(copied_path)

# Drop first unnamed column if it's an index
data.drop(data.columns[0], inplace=True, axis=1)

# Strip whitespace from column names and string entries
data.columns = data.columns.str.strip()
data = data.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# Reordering columns
cols = list(data.columns.values)
cols.remove('City')
cols.remove('PCOS')
cols.remove('PCOS_from')
data = data[cols + ['City', 'PCOS_from', 'PCOS']]

# Map target to numeric
data['PCOS'] = data['PCOS'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])

# Convert the DataFrame to a clean list of lists
dataset = data.values.tolist()

def loadDataset(dataset, split, trainingSet=[], testSet=[]):
    for row in dataset:
        # Convert all values to int or float
        clean_row = []
        for val in row:
            try:
                clean_row.append(float(val))
            except:
                clean_row.append(0)
        if random.random() < split:
            trainingSet.append(clean_row)
        else:
            testSet.append(clean_row)

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return correct / float(len(testSet))

def main():
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(dataset, split, trainingSet, testSet)

    print('Train set:', len(trainingSet))
    print('Test set:', len(testSet))

    k = 3

    # Predict on training set
    predictions_train = []
    for x in range(len(trainingSet)):
        neighbors = getNeighbors(trainingSet, trainingSet[x], k)
        result = getResponse(neighbors)
        predictions_train.append(result)
    accuracy_train = getAccuracy(trainingSet, predictions_train)
    print('Training Accuracy:', accuracy_train)

    # Predict on test set
    predictions_test = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions_test.append(result)
    accuracy_test = getAccuracy(testSet, predictions_test)
    print('Test Accuracy:', accuracy_test)

    actual = [row[-1] for row in testSet]
    print('Confusion Matrix:\n', confusion_matrix(actual, predictions_test))
    print('Accuracy Score:', accuracy_score(actual, predictions_test))
    print('Classification Report:\n', classification_report(actual, predictions_test))

    # Save model (training set and k value)
    model_data = {
        "trainingSet": trainingSet,
        "k": k
    }
    directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
    os.makedirs(directory, exist_ok=True)
    filename = f'{directory}/knn_model.pkl'
    pickle.dump(model_data, open(filename, 'wb'))

main()

def predict(trainingSet, X_test, k):
    predictions = []
    for test_instance in X_test:
        neighbors = getNeighbors(trainingSet, test_instance, k)
        result = getResponse(neighbors)
        predictions.append(result)
    return predictions
