# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import os
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier


# # Load and clean data
# path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
# data = pd.read_csv(path)
# data.columns = data.columns.str.strip()

# # Drop unnecessary columns
# del data['PCOS_from']
# del data['City']
# del data['relocated city']

# # Label encoding
# data['PCOS_label'] = None
# data = data.set_index('PCOS_label').reset_index()

# def label(row):
#     return 1 if row['PCOS'].strip().lower() == 'yes' else 0
# data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)
# print("Class distribution after labeling:", data['PCOS_label'].value_counts(), '\n')

# # Label mapping
# PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))

# # Feature selection
# X = data.drop(['PCOS_label', 'PCOS'], axis=1)
# y = data['PCOS_label']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# print("Class distribution in y_train:", np.bincount(y_train))
# print("Class distribution in y_test:", np.bincount(y_test), '\n')

# # Random Forest model
# rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=0)
# rf_clf.fit(X_train, y_train)

# # Predictions and Evaluation
# rf_predictions = rf_clf.predict(X_test)

# print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions), '\n')
# print("Accuracy: {:.2f}".format(accuracy_score(y_test, rf_predictions)))
# print("Precision: {:.2f}".format(precision_score(y_test, rf_predictions)))
# print("Recall: {:.2f}".format(recall_score(y_test, rf_predictions)))
# print("F1 Score: {:.2f}".format(f1_score(y_test, rf_predictions)), '\n')

# print("Classification Report:\n", classification_report(y_test, rf_predictions, target_names=['No', 'Yes']))
# print("Training Accuracy: {:.2f}".format(rf_clf.score(X_train, y_train)))
# print("Testing Accuracy: {:.2f}".format(rf_clf.score(X_test, y_test)))

# # Save the model
# directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
# os.makedirs(directory, exist_ok=True)

# filename = f'{directory}/rf_model.pkl'
# pickle.dump(rf_clf, open(filename, 'wb'))

# # Test prediction example
# test_data = pd.DataFrame([[5, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 3, 3, 0]], columns=X.columns)
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.predict(test_data)
# print("Prediction Example:", PCOS_check[result[0]])

# # Feature importance plot
# importances = rf_clf.feature_importances_
# importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
# plt.xlabel('Feature Importance')
# plt.title('Random Forest Feature Importances')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# file_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Final Model - Best Accuracy/feature_importance_rf.png"
# plt.savefig(file_path, dpi=300, bbox_inches='tight')
# plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve, auc, roc_curve
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load and clean data
path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(path)
data.columns = data.columns.str.strip()

# Drop unnecessary columns
del data['PCOS_from']
del data['City']
del data['relocated city']

# Label encoding
data['PCOS_label'] = None
data = data.set_index('PCOS_label').reset_index()

def label(row):
    return 1 if row['PCOS'].strip().lower() == 'yes' else 0
data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)

PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))

# Feature selection
X = data.drop(['PCOS_label', 'PCOS'], axis=1)
y = data['PCOS_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=0, n_neighbors=5)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# Random Forest model with class_weight='balanced'
rf_clf = RandomForestClassifier(random_state=0)

# Grid Search to tune hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [5, 10, 15, None],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples for splitting
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'class_weight': ['balanced', None],  # Handle class imbalance
    'bootstrap': [True, False],  # Use bootstrap sampling or not
    'max_features': ['sqrt', 'log2'],  # Explore feature subsets
    'min_impurity_decrease': [0.0, 0.01, 0.1]  # Regularization for deeper trees
}

grid_search = GridSearchCV(rf_clf, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)

# Best parameters found by GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}\n")

# Retraining with the best parameters
best_rf_clf = grid_search.best_estimator_

# Cross-validation score (for better performance estimate)
cv_scores = cross_val_score(best_rf_clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")

# Predictions and Evaluation
rf_predictions = best_rf_clf.predict(X_test)

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions), '\n')
print("Accuracy: {:.2f}".format(accuracy_score(y_test, rf_predictions)))
print("Precision: {:.2f}".format(precision_score(y_test, rf_predictions)))
print("Recall: {:.2f}".format(recall_score(y_test, rf_predictions)))
print("F1 Score: {:.2f}".format(f1_score(y_test, rf_predictions)), '\n')

# Classification report
print("Classification Report:\n", classification_report(y_test, rf_predictions, target_names=['No', 'Yes']))

# Training and Testing Accuracy
print("Training Accuracy: {:.2f}".format(best_rf_clf.score(X_train_res, y_train_res)))
print("Testing Accuracy: {:.2f}".format(best_rf_clf.score(X_test, y_test)))

# Save the model
directory = 'C:/Users/hp/Desktop/Project/PCOS Prediction/Model'
os.makedirs(directory, exist_ok=True)

filename = f'{directory}/rf_model_best.pkl'
pickle.dump(best_rf_clf, open(filename, 'wb'))

# Test prediction example
test_data = pd.DataFrame([[5, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 3, 3, 0]], columns=X.columns)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(test_data)
print("Prediction Example:", PCOS_check[result[0]])

# Feature importance plot
importances = best_rf_clf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()

file_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Final Model - Best Accuracy/feature_importance_rf_best.png"
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_rf_clf.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ROC Curve
y_probs = best_rf_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
