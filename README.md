# diabetes-prediction-model

# 🩺 Diabetes Prediction System

This project is a Machine Learning application that predicts whether a patient has diabetes based on diagnostic measurements. The model utilizes **Logistic Regression** and demonstrates the importance of **Feature Scaling (Standardization)** in classification problems.

## 🚀 Project Overview
The goal is to classify patients into two groups:
* **0 (Negative):** Healthy / No Diabetes
* **1 (Positive):** Diabetic

Using the Pima Indians Diabetes Dataset, I built a predictive model that achieves **~74.68% accuracy**.

## 📊 Dataset Features
The dataset consists of medical predictors and one target variable:
* **Pregnancies:** Number of times pregnant
* **Glucose:** Plasma glucose concentration
* **BloodPressure:** Diastolic blood pressure (mm Hg)
* **SkinThickness:** Triceps skin fold thickness (mm)
* **Insulin:** 2-Hour serum insulin (mu U/ml)
* **BMI:** Body mass index
* **DiabetesPedigreeFunction:** Diabetes pedigree function
* **Age:** Age (years)
* **Outcome:** Class variable (0 or 1)

## 🛠️ Methodology & Process

### 1. Exploratory Data Analysis (EDA)
* Analyzed the relationship between `Glucose` levels and the `Outcome`.
* Checked for class imbalance and data distribution.

### 2. Model Evolution
I followed an iterative approach to improve the model's performance:

* **Iteration 1 (Base Model):** Trained Logistic Regression on raw data.
    * *Result:* Accuracy ~74.03%
    * *Observation:* The model struggled with convergence due to varying scales of features (e.g., Insulin vs. Age).
    
* **Iteration 2 (Optimized Model):** Applied **StandardScaler** to normalize features (Mean=0, Std=1).
    * *Result:* Accuracy increased to **74.68%**
    * *Benefit:* The optimization process became stable, and `ConvergenceWarning` was resolved.

## 📈 Model Performance
* **Algorithm:** Logistic Regression (max_iter=1000)
* **Final Accuracy:** 74.68%
* **Confusion Matrix Analysis:**
    * **True Negatives (TN):** 90 (Healthy patients correctly identified)
    * **False Negatives (FN):** 29 (Missed cases - Type 2 Error)
    * **False Positives (FP):** 10 (False Alarms - Type 1 Error)
    * **True Positives (TP):** 25 (Diabetic patients correctly identified)

## 💻 Technologies Used
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib** (Visualization)
* **Scikit-Learn** (Model, Scaling, Metrics)

## 📥 Installation & Usage

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/diabetes-prediction.git](https://github.com/YOUR_USERNAME/diabetes-prediction.git)
    ```
2.  **Install requirements:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
3.  **Run the Notebook:**
    Open `diabetes_prediction.ipynb` to see the step-by-step analysis.

---
## 👨‍💻 Author
**Developed by [Ergün Üngör](https://github.com/ergunungor)**
