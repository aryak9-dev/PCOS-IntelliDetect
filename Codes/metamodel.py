import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from knnprediction import predict

# Load your KNN model
with open('Model/knn_model.pkl', 'rb') as f:
    knn_model_data = pickle.load(f)

trainingSet = knn_model_data['trainingSet']
k = knn_model_data['k']

# Predict using the imported function
# custom_knn_pred = predict(trainingSet, X_test.values.tolist(), k)

# === Load All Trained Models ===
with open('Model/model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

# with open('Model/knn_model.pkl', 'rb') as f:
#     knn_model = pickle.load(f)
# with open('Model/knn_model.pkl', 'rb') as f:
#     knn_model = pickle.load(f)
#     knn_model = knn_model['model']  # <- this line fixes the error


with open('Model/Logistic_Regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('Model/Custom_NB_model.pkl', 'rb') as f:
    custom_nb_model = pickle.load(f)

with open('Model/NB_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('Model/rf_model_best.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('Model/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# === Load and Clean Data ===
path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data = pd.read_csv(path)
data.columns = data.columns.str.strip()

# Drop unused columns
for col in ['PCOS_from', 'City', 'relocated city']:
    if col in data.columns:
        del data[col]

# Encode target variable
data['PCOS_label'] = data['PCOS'].apply(lambda val: 1 if val.strip().lower() == 'yes' else 0)

# Features and target
X = data.drop(['PCOS', 'PCOS_label'], axis=1)
y = data['PCOS_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# === Get Predictions from Each Model ===
dt_pred = dt_model.predict(X_test)
# knn_pred = knn_model.predict(X_test)
knn_pred = predict(trainingSet, X_test.values.tolist(), k)
lr_pred = lr_model.predict(X_test)
custom_nb_pred = custom_nb_model.predict(X_test)
nb_pred = nb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# === Stack Predictions for Meta-Model ===
stacked_preds = np.column_stack((dt_pred, knn_pred, lr_pred, custom_nb_pred, nb_pred, rf_pred, xgb_pred))

# === Train Meta-Model (Logistic Regression) ===
meta_model = LogisticRegression()
meta_model.fit(stacked_preds, y_test)

# === Final Prediction and Evaluation ===
final_predictions = meta_model.predict(stacked_preds)

print("✅ Ensemble Model Accuracy:", accuracy_score(y_test, final_predictions))
print("\n📋 Classification Report:\n", classification_report(y_test, final_predictions))

# === Save Meta-Model ===
with open('Model/meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
    print("🧠 Meta-model saved as Model/meta_model.pkl")
