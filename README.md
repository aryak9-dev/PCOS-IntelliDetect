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

## 📁 Project Structure

PCOS-IntelliDetect/
│── dataset/
│── codes/
│── model/model.pkl
│── scripts/
│── README.md

---

## ▶️ How to Run

```bash
git clone https://github.com/aryak9-dev/PCOS-IntelliDetect.git
cd PCOS-IntelliDetect

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

