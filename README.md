# PCOS-IntelliDetect

# 🩺 PCOS IntelliDetect

A Machine Learning-based system for predicting Polycystic Ovary Syndrome (PCOS) using clinical and health-related parameters. The project builds a complete ML pipeline and uses a Decision Tree model for classification.

---

## 🚀 Overview

PCOS IntelliDetect predicts whether a patient is likely to have PCOS based on medical attributes. The system processes structured data, trains a classification model, and evaluates its performance.

---

## 🎯 Objectives

- Predict PCOS using patient health data  
- Build a complete machine learning pipeline  
- Evaluate model performance using standard metrics  

---

## 🧠 Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Model:** Decision Tree Classifier  
- **Environment:** Jupyter Notebook  

---

## 📊 Dataset

- **Total Records:** 268 patient entries  
- **Features Used:** 26 clinical attributes  
- **Target Variable:** PCOS_label (binary classification)  

---

## ⚙️ Machine Learning Approach

- Performed data preprocessing and feature selection  
- Split dataset into training and testing sets using `train_test_split`  
- Trained a Decision Tree model with `max_depth = 6`  
- Evaluated model using confusion matrix  
- Achieved approximately **87% accuracy**  

---

## 🔄 Workflow

1. Data Loading  
2. Data Cleaning & Preprocessing  
3. Feature Selection  
4. Train-Test Split  
5. Model Training (Decision Tree)  
6. Model Evaluation  
7. Prediction  

---

## 🌳 Final Model

The final model used in this project is a:

👉 **Decision Tree Classifier**

- Selected based on performance and interpretability  
- Provides clear decision rules for prediction  
- Helps in understanding feature importance  

---

## 📈 Feature Importance Insights

From the trained model:

- **Overweight** is the most important feature  
- Followed by:
  - Period Length  
  - Irregular or missed periods  
  - Exercise frequency  

👉 This shows lifestyle and menstrual patterns are key indicators in prediction  

---

## 🌐 Web Interface

- The frontend interface is built using HTML and CSS  
- The main entry page:

👉 **`index.html` is located inside the `templates/` folder**

- It takes user input and sends it to the model for prediction  

---
## 📈 Key Insights from Model

- Overweight is the most influential factor in predicting PCOS  
- Menstrual cycle-related features (irregular periods, cycle length) are strong predictors  
- Lifestyle factors such as exercise and eating habits also contribute to prediction  
- Physical symptoms like hair growth and acne have moderate importance  
- Some features such as mood swings and fatigue have minimal impact on the model  

These insights align with real-world understanding of PCOS and highlight the importance of lifestyle and hormonal indicators.

---

## 📁 Project Structure

PCOS-IntelliDetect/
│── dataset/
│── codes/
│── model/model.pkl
│── templates/index.html
│── README.md

---

## ▶️ How to Run


```bash
git clone https://github.com/aryak9-dev/PCOS-IntelliDetect.git
cd PCOS-IntelliDetect

python3 -m venv venv
source venv/bin/activate
python app.py

