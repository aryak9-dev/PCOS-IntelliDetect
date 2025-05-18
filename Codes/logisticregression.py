# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# import numpy as np
# import os
# import pickle

# data_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
# data = pd.read_csv(data_path)
# data.columns = data.columns.str.strip()
# data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# data.head()

# del data['PCOS_from']
# del data['City']
# del data['relocated city']

# data.head()

# data['PCOS_label'] = None
# data = data.set_index('PCOS_label')
# data = data.reset_index()
# data.head()


# def label(row):
#     if row['PCOS'] == 'Yes':
#         return 1
#     else:
#         return 0

# data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)

# data.head()

# PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))
# PCOS_check

# x = data.drop(['PCOS_label', 'PCOS'], axis=1)
# y = data.PCOS_label

# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

# sc_x = StandardScaler()
# xtrain = sc_x.fit_transform(xtrain)
# xtest = sc_x.transform(xtest)

# # print(xtrain[0:10, :], '\n')

# classifier = LogisticRegression(random_state=0)
# classifier.fit(xtrain, ytrain)

# y_pred = classifier.predict(xtest)

# cm = confusion_matrix(ytest, y_pred)

# print("Confusion Matrix : \n", cm)

# print("Accuracy : ", accuracy_score(ytest, y_pred))

# print(classification_report(
#     ytest, y_pred, target_names=['No', 'Yes']))

# directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
# os.makedirs(directory, exist_ok=True)

# filename = f'{directory}/Custom_NB_model.pkl'
# pickle.dump(model, open(filename, 'wb'))


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import pickle

data_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

data.head()

# Data Preprocessing
del data['PCOS_from']
del data['City']
del data['relocated city']

data.head()

# Create new label column for 'PCOS'
data['PCOS_label'] = None
data = data.set_index('PCOS_label')
data = data.reset_index()
data.head()

# Function to map 'Yes'/'No' to 1/0
def label(row):
    if row['PCOS'] == 'Yes':
        return 1
    else:
        return 0

# Apply label function
data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)

# Check the mapping for PCOS
PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))
PCOS_check

# Features (X) and target (y)
x = data.drop(['PCOS_label', 'PCOS'], axis=1)
y = data.PCOS_label

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

# Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)

# Predictions
y_pred = classifier.predict(xtest)

# Confusion Matrix and Evaluation Metrics
cm = confusion_matrix(ytest, y_pred)

print("Confusion Matrix : \n", cm)
print("Accuracy : ", accuracy_score(ytest, y_pred))
print(classification_report(
    ytest, y_pred, target_names=['No', 'Yes']))

# Save the model using pickle
model_filename = 'logistic_regression_model.pkl'

directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
os.makedirs(directory, exist_ok=True)

filename = f'{directory}/Logistic_Regression_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))