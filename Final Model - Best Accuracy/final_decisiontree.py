from six import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
import pickle

path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(path)
data.columns = data.columns.str.strip()

data.head()

del data['PCOS_from']

del data['City']

del data['relocated city']

data['PCOS_label'] = None
data.head()

data = data.set_index('PCOS_label')

data = data.reset_index()

def label(row):
    return 1 if row['PCOS'].strip().lower() == 'yes' else 0
data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)
print("Class distribution after labeling:", data['PCOS_label'].value_counts())
print('\n')

data.head()

PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))
PCOS_check

X = data.drop(['PCOS_label', 'PCOS'], axis=1)
y = data.PCOS_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=0)

X_train.head()
y_train.head()
print("Class distribution in y_train:", np.bincount(y_train))
print("Class distribution in y_test:", np.bincount(y_test))
print('\n')

clf = DecisionTreeClassifier(max_depth=6).fit(X_train, y_train)

tree_predicted = clf.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)
print('Confusion Matrix: \n', confusion, '\n')

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
print('\n')

print(classification_report(
    y_test, tree_predicted, target_names=['No', 'Yes']))

print('Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))

feature_cols = ['Period Length', 'Cycle Length', 'Age', 'Overweight', 
                'loss weight gain / weight loss', 'irregular or missed periods', 
                'Difficulty in conceiving', 'Hair growth on Chin', 'Hair growth on Cheeks', 
                'Hair growth Between breasts','Hair growth on Upper lips', 'Hair growth in Arms', 
                'Hair growth on Inner thighs', 'Acne or skin tags', 'Hair thinning or hair loss ', 
                'Dark patches', 'always tired', 'more Mood Swings', 'exercise per week', 
                'eat outside per week', 'canned food often']

pcos_prediction = clf.predict(X_test)
PCOS_check[pcos_prediction[0]]

pcos1 = clf.predict(pd.DataFrame([[5, 6, 2, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 7, 0]], columns= X.columns))
print(PCOS_check[pcos1[0]])

pcos2 = clf.predict(pd.DataFrame([[5, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 3, 3, 0]], columns= X.columns))
print(PCOS_check[pcos2[0]])

directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
os.makedirs(directory, exist_ok=True)

filename = f'{directory}/model.pkl'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
test_data = pd.DataFrame([[5, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 3, 3, 0]], columns=X.columns)
result = loaded_model.predict(test_data)
print(PCOS_check[result[0]]) 

import matplotlib.pyplot as plt

feature_importances = clf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Decision Tree Model')
plt.gca().invert_yaxis() 

file_path = "C:/Users/hp/Desktop/Project/PCOS Prediction\Final Model - Best Accuracy/feature_importance.png" 
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.show()


