# 📊 Customer Churn Prediction & AI-Powered Insights

An end-to-end **Machine Learning application** that predicts whether a customer is likely to churn, assigns a **churn-risk level**, and uses **Google Gemini AI** to explain the prediction and suggest practical customer-retention strategies.

The project covers the complete ML lifecycle — from **EDA and feature engineering to model optimization, deployment, and Generative AI integration**.

---

## 🚀 Project Overview

Customer churn occurs when customers stop using a company's products or services.

This project analyzes customer **demographic, usage, support, payment, and contract information** to predict churn probability and identify customers who may be at risk of leaving.

The application:

* 📈 Predicts **customer churn probability**
* 🚦 Classifies customers into **Low, Medium, or High risk**
* 🤖 Compares multiple Machine Learning models
* ⚙️ Optimizes the best-performing models using `RandomizedSearchCV`
* 💾 Saves and reloads the trained model and preprocessing pipeline
* 🧠 Generates AI-powered explanations using **Google Gemini**
* 💡 Provides practical **customer-retention recommendations**
* 🌐 Deploys the complete solution through a **Streamlit web application**

---

## 🎯 Objectives

The main objectives of this project are to:

1. Understand customer churn patterns through **Exploratory Data Analysis (EDA)**
2. Perform **data cleaning and feature engineering**
3. Build a reusable **data preprocessing pipeline**
4. Train and compare multiple classification models
5. Evaluate models using multiple performance metrics
6. Optimize promising models using **RandomizedSearchCV**
7. Save the final model and preprocessing pipeline
8. Build an interactive **Streamlit application**
9. Integrate **Google Gemini** for natural-language explanations
10. Provide actionable customer-retention suggestions

---

## 🧠 End-to-End Workflow

```text
                    ┌──────────────────────┐
                    │   Raw Customer Data  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Data Cleaning     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Preprocessing     │
                    │  Encoding + Scaling  │
                    └──────────┬───────────┘
                               ↓
              ┌─────────────────────────────────┐
              │       Model Training            │
              │ Logistic Regression             │
              │ Random Forest                   │
              │ XGBoost                         │
              └───────────────┬─────────────────┘
                              ↓
                    ┌──────────────────────┐
                    │ Model Comparison     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ RandomizedSearchCV   │
                    │ Hyperparameter Tuning│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Tuned XGBoost Model │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Save Model + Pipeline│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Streamlit Web App    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Churn Probability    │
                    │ Risk Classification  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Google Gemini AI     │
                    │ Explanation + Actions │
                    └──────────────────────┘
```

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_xgboost_model.pkl
├── churn_preprocessor.pkl
├── requirements.txt
└── README.md
```

### 📄 File Description

| File                      | Description                            |
| ------------------------- | -------------------------------------- |
| `app.py`                  | Streamlit application                  |
| `churn_xgboost_model.pkl` | Trained XGBoost churn prediction model |
| `churn_preprocessor.pkl`  | Saved preprocessing pipeline           |
| `requirements.txt`        | Required Python dependencies           |
| `README.md`               | Project documentation                  |

---

## 📊 Dataset

The dataset contains customer information covering:

* 👤 Demographics
* 📱 Product/service usage
* ☎️ Customer support interactions
* 💳 Payment behavior
* 📋 Subscription information
* 📅 Contract details
* 💰 Customer spending
* 🔄 Customer churn

### 📌 Features

| Feature             | Description                                |
| ------------------- | ------------------------------------------ |
| `CustomerID`        | Unique customer identifier                 |
| `Age`               | Customer age                               |
| `Gender`            | Customer gender                            |
| `Tenure`            | Duration of customer relationship          |
| `Usage Frequency`   | Frequency of product/service usage         |
| `Support Calls`     | Number of customer support calls           |
| `Payment Delay`     | Payment delay information                  |
| `Subscription Type` | Basic, Standard, or Premium                |
| `Contract Length`   | Monthly, Quarterly, or Annual              |
| `Total Spend`       | Total amount spent                         |
| `Last Interaction`  | Days since the customer's last interaction |
| `Churn`             | Target variable                            |

### 🎯 Target Variable

```text
0 → Customer is not likely to churn
1 → Customer is likely to churn
```

---

## 🛠️ Feature Engineering

Additional features were created to provide the models with more meaningful information about customer activity and behavior.

### 1. 📈 Total Activity

Combines product usage and customer support interactions.

```text
Total Activity = Usage Frequency + Support Calls
```

### 2. ☎️ Support Calls per Tenure

Measures support-call frequency relative to the customer's relationship duration.

```text
Support Calls per Tenure =
Support Calls / (Tenure + 1)
```

### 3. 💳 Payment Delay per Tenure

Measures payment delay relative to customer tenure.

```text
Payment Delay per Tenure =
Payment Delay / (Tenure + 1)
```

The `+1` prevents division by zero.

---

## ⚙️ Data Preprocessing

A reusable preprocessing pipeline was implemented using **Scikit-learn's `ColumnTransformer`**.

The preprocessing workflow includes:

* 🗑️ Removing the `CustomerID` identifier
* 🧹 Handling missing target values
* 🔤 One-hot encoding categorical variables
* 📏 Standardizing numerical features
* 🔄 Keeping preprocessing steps inside a reusable pipeline

The preprocessing pipeline is saved as:

```text
churn_preprocessor.pkl
```

This ensures that the **same preprocessing logic used during training is also used during deployment**.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared:

### 1. Logistic Regression

A linear classification model used as a strong baseline.

### 2. Random Forest

An ensemble of decision trees capable of capturing nonlinear relationships between customer characteristics and churn.

### 3. XGBoost

A powerful gradient-boosting algorithm designed to perform well on structured/tabular data.

---

## ⚙️ Hyperparameter Optimization

The strongest-performing models were further optimized using:

**`RandomizedSearchCV`**

Configuration:

| Setting             | Value              |
| ------------------- | ------------------ |
| Cross-validation    | 3-Fold             |
| Optimization Metric | F1 Score           |
| Search Method       | RandomizedSearchCV |

Using F1 score helps balance **precision and recall**, which is particularly useful when identifying customers who may churn.

---

## 📈 Model Performance

### Final Model Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.8935 |     0.9235 |     0.8856 |     0.9041 |     0.9590 |
| Random Forest       |     0.9996 |     0.9999 |     0.9993 |     0.9996 |     1.0000 |
| Tuned Random Forest |     0.9999 |     0.9999 |     0.9999 |     0.9999 |     1.0000 |
| XGBoost             |     0.9999 |     1.0000 |     0.9999 |     0.9999 |     1.0000 |
| **Tuned XGBoost**   | **0.9999** | **1.0000** | **0.9999** | **0.9999** | **1.0000** |

### 🏆 Final Selected Model

The **Tuned XGBoost** model was selected as the final model.

It is saved as:

```text
churn_xgboost_model.pkl
```

---

## 🔍 Important Features

Feature-importance analysis identified several important predictors, including:

* ☎️ Support Calls
* 💰 Total Spend
* 👤 Age
* 💳 Payment Delay
* 📋 Contract Length
* 📞 Support Calls per Tenure
* 🕐 Last Interaction
* 📅 Tenure

These features help the model identify behavioral and customer-profile patterns associated with churn.

---

# 🌐 Streamlit Web Application

The final Machine Learning pipeline is integrated into an interactive **Streamlit application**.

Users can enter customer information and receive an instant churn prediction.

---

## 👤 Customer Information

The application accepts:

* Age
* Gender
* Tenure
* Subscription Type
* Contract Length

## 📈 Usage Information

* Usage Frequency
* Support Calls
* Last Interaction

## 💳 Payment Information

* Payment Delay
* Total Spend

---

## 📊 Application Output

After clicking **Predict Churn**, the application displays:

### 🎯 Churn Probability

The estimated probability that the customer will churn.

### 🚦 Risk Segment

The customer is categorized into:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

### 🔮 Churn Prediction

A final prediction indicating whether the customer is likely to churn.

### 👤 Customer Summary

A concise summary of the provided customer information.

### 🧠 AI-Powered Explanation

Google Gemini explains the prediction in simple, human-readable language.

### 💡 Retention Suggestions

The AI provides practical actions that could potentially help reduce customer churn.

---

# 🚦 Churn Risk Segmentation

Customers are categorized according to their predicted churn probability.

| Churn Probability | Risk Segment   |
| ----------------: | -------------- |
|          `< 0.30` | 🟢 Low Risk    |
|     `0.30 – 0.70` | 🟡 Medium Risk |
|          `> 0.70` | 🔴 High Risk   |

This converts a numerical probability into an easier-to-understand business risk category.

---

# 🧠 Google Gemini Integration

The application integrates **Google Gemini** to provide an AI-generated explanation of each prediction.

Instead of showing only:

```text
Churn Probability: 87%
```

the application can provide a natural-language explanation such as:

```text
The customer has a high churn risk. Frequent support calls,
payment delays, and recent interaction patterns may indicate
potential dissatisfaction or engagement issues.
```

### Gemini receives a limited customer summary containing:

* Churn probability
* Risk segment
* Churn prediction
* Selected customer features

Gemini is instructed to:

1. Explain why the customer has the given risk level
2. Identify important factors visible in the provided data
3. Suggest practical customer-retention actions
4. Avoid inventing information that is not present in the provided data


# 🧪 Complete ML Pipeline

The project follows an end-to-end production-style workflow:

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Hyperparameter Optimization
        ↓
Final Tuned XGBoost
        ↓
Save Model + Preprocessor
        ↓
Streamlit Deployment
        ↓
Customer Input
        ↓
Churn Probability
        ↓
Risk Segmentation
        ↓
Gemini AI Explanation
        ↓
Retention Recommendations
```

---

# ⚠️ Model Limitations & Important Considerations

Although the tree-based models achieved extremely high evaluation scores, these results should be interpreted carefully.

The dataset shows **strong feature separation**, including a particularly strong relationship between contract type and churn.

Therefore, the reported performance may not represent how the model would perform on a more diverse real-world customer population.

### For production use, the model should be:

* 🧪 Validated on an independent real-world dataset
* 🔄 Tested using additional unseen customer populations
* 📊 Monitored after deployment
* ⚖️ Evaluated for potential data leakage
* 📈 Re-evaluated periodically as customer behavior changes

High validation performance does not automatically guarantee strong real-world generalization.

---

# 🧰 Technologies Used

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| 🐍 Python             | Core programming language            |
| 🐼 Pandas             | Data manipulation                    |
| 🔢 NumPy              | Numerical computing                  |
| 🤖 Scikit-learn       | Preprocessing, modeling & evaluation |
| 🌳 XGBoost            | Gradient boosting model              |
| 💾 Joblib             | Model serialization                  |
| 📊 Streamlit          | Web application                      |
| 🧠 Google Gemini      | AI-powered explanations              |
| ⚙️ RandomizedSearchCV | Hyperparameter optimization          |

---

# 💡 Key Project Highlights

This project demonstrates practical experience with:

* ✅ Exploratory Data Analysis
* ✅ Feature Engineering
* ✅ Data Preprocessing
* ✅ Classification Algorithms
* ✅ Model Evaluation
* ✅ Hyperparameter Optimization
* ✅ Scikit-learn Pipelines
* ✅ XGBoost
* ✅ Model Serialization
* ✅ Streamlit Deployment
* ✅ Generative AI Integration
* ✅ API Key Management
* ✅ AI-assisted Business Insights

