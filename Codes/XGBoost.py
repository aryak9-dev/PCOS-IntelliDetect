from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Load and clean dataset
path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(path)
data.columns = data.columns.str.strip()

# Drop unnecessary columns
data.drop(['PCOS_from', 'City', 'relocated city'], axis=1, inplace=True)

# Encode label
def label(row):
    return 1 if row['PCOS'].strip().lower() == 'yes' else 0

data['PCOS_label'] = data.apply(lambda row: label(row), axis=1)

print("Class distribution after labeling:", data['PCOS_label'].value_counts())
print('\n')

PCOS_check = dict(zip(data.PCOS_label.unique(), data.PCOS.unique()))

X = data.drop(['PCOS_label', 'PCOS'], axis=1)
y = data['PCOS_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Class distribution in y_train:", np.bincount(y_train))
print("Class distribution in y_test:", np.bincount(y_test))
print('\n')

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_test)

# Metrics
conf = confusion_matrix(y_test, xgb_pred)
acc = accuracy_score(y_test, xgb_pred)
prec = precision_score(y_test, xgb_pred)
rec = recall_score(y_test, xgb_pred)
f1 = f1_score(y_test, xgb_pred)

print('Confusion Matrix:\n', conf, '\n')
print('Accuracy: {:.2f}'.format(acc))
print('Precision: {:.2f}'.format(prec))
print('Recall: {:.2f}'.format(rec))
print('F1 Score: {:.2f}'.format(f1))
print('\n')

print("Classification Report:\n", classification_report(y_test, xgb_pred, target_names=['No', 'Yes']))

# Accuracy on train and test
print("Training Accuracy:", xgb_model.score(X_train, y_train))
print("Testing Accuracy:", xgb_model.score(X_test, y_test))

# Save model
model_dir = "C:/Users/hp/Desktop/Project/PCOS Prediction/Model"
os.makedirs(model_dir, exist_ok=True)
pickle.dump(xgb_model, open(f"{model_dir}/xgb_model.pkl", "wb"))

# Sample prediction
sample = pd.DataFrame([[5, 1, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 3, 3, 0]], columns=X.columns)
result = xgb_model.predict(sample)
print("Prediction Example:", PCOS_check[result[0]])

# Feature Importance Plot
plt.figure(figsize=(10, 6))
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save plot
plt.savefig("C:/Users/hp/Desktop/Project/PCOS Prediction/Final Model - Best Accuracy/xgb_feature_importance.png", dpi=300)
plt.show()
