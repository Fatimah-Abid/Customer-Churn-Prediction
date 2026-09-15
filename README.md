An end-to-end Machine Learning application that predicts whether a customer is likely to churn and provides an AI-powered explanation of the prediction using Google Gemini.

🚀 Project Overview

Customer churn occurs when customers stop using a company's products or services.

This project uses customer demographic, usage, support, payment, and contract information to:

Predict customer churn probability
Classify customers into Low, Medium, or High churn-risk segments
Compare multiple Machine Learning models
Optimize the best-performing models using RandomizedSearchCV
Provide an AI-generated explanation of the prediction
Suggest practical customer-retention actions

The final solution is deployed through a Streamlit web application.

🎯 Objectives

The main objectives of this project are:

Understand customer churn patterns through EDA.
Perform feature engineering and preprocessing.
Train multiple Machine Learning classification models.
Compare model performance using multiple evaluation metrics.
Optimize the best-performing models.
Save and reload the trained model and preprocessing pipeline.
Build an interactive Streamlit application.
Integrate Google Gemini for plain-language churn explanations.
📂 Project Structure
customer-churn-prediction/
│
├── app.py
├── churn_xgboost_model.pkl
├── churn_preprocessor.pkl
├── requirements.txt
└── README.md
File Description
File	Description
app.py	Streamlit application
churn_xgboost_model.pkl	Trained XGBoost churn prediction model
churn_preprocessor.pkl	Saved preprocessing pipeline
requirements.txt	Required Python packages
README.md	Project documentation
📊 Dataset

The dataset contains customer information related to demographics, usage, support interactions, payments, subscriptions, and churn.

Features
Feature	Description
CustomerID	Unique customer identifier
Age	Customer age
Gender	Customer gender
Tenure	Duration of customer relationship
Usage Frequency	Frequency of product/service usage
Support Calls	Number of support calls
Payment Delay	Payment delay information
Subscription Type	Basic, Standard, or Premium
Contract Length	Monthly, Quarterly, or Annual
Total Spend	Total amount spent
Last Interaction	Days since last interaction
Churn	Target variable
Target Variable
0 → Customer is not likely to churn
1 → Customer is likely to churn
🛠️ Feature Engineering

Additional features were created to improve the model:

Total Activity
Total Activity = Usage Frequency + Support Calls
Support Calls per Tenure
Support Calls per Tenure =
Support Calls / (Tenure + 1)
Payment Delay per Tenure
Payment Delay per Tenure =
Payment Delay / (Tenure + 1)
⚙️ Data Preprocessing

The preprocessing pipeline includes:

Removing the CustomerID identifier
Handling missing target values
One-hot encoding categorical variables
Standardizing numerical features
Keeping preprocessing inside a reusable ColumnTransformer

The preprocessing pipeline is saved as:

churn_preprocessor.pkl

This allows the exact same preprocessing steps to be used during deployment.

🤖 Machine Learning Models

Three baseline models were trained and compared:

Logistic Regression
Random Forest
XGBoost

The best-performing models were further optimized using:

RandomizedSearchCV

with 3-fold cross-validation and F1 score as the optimization metric.

📈 Model Performance
Final Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Logistic Regression	0.8935	0.9235	0.8856	0.9041	0.9590
Random Forest	0.9996	0.9999	0.9993	0.9996	1.0000
Tuned Random Forest	0.9999	0.9999	0.9999	0.9999	1.0000
XGBoost	0.9999	1.0000	0.9999	0.9999	1.0000
Tuned XGBoost	0.9999	1.0000	0.9999	0.9999	1.0000
Final Selected Model

Tuned XGBoost

The tuned XGBoost model was selected as the final model and saved as:

churn_xgboost_model.pkl
🔍 Important Features

Feature-importance analysis identified several important predictors, including:

Support Calls
Total Spend
Age
Payment Delay
Contract Length
Support Calls per Tenure
Last Interaction
Tenure

These features help the model identify patterns associated with customer churn.

🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can enter:

👤 Customer Information
Age
Gender
Tenure
Subscription Type
Contract Length
📈 Usage Information
Usage Frequency
Support Calls
Last Interaction
💳 Payment Information
Payment Delay
Total Spend

After clicking Predict Churn, the application displays:

Churn probability
Risk segment
Churn prediction
Customer summary
AI-powered explanation
Customer-retention suggestions
🚦 Risk Segmentation

Customers are divided into three risk categories based on predicted churn probability:

Probability	Risk Segment
< 0.30	🟢 Low
0.30 – 0.70	🟡 Medium
> 0.70	🔴 High
🧠 AI-Powered Explanation

The application integrates Google Gemini to provide a simple explanation of the Machine Learning prediction.

The application sends a limited customer summary containing:

Churn probability
Risk segment
Prediction
Selected customer features

Gemini is instructed to:

Explain why the customer has the given risk level.
Identify important factors visible in the provided data.
Suggest practical customer-retention actions.
Avoid inventing information.

The Gemini API key is not stored in the source code.

🔐 Environment Variable

For local development, configure the Gemini API key as an environment variable:

GEMINI_API_KEY=your_api_key_here

Never commit the actual API key to GitHub.

💻 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git

Move into the project directory:

cd CustomerChurn

Install dependencies:

pip install -r requirements.txt

Configure the Gemini API key.

Then run the Streamlit application:

streamlit run app.py

The application will open in your browser.

☁️ Deployment

This application can be deployed using Streamlit Community Cloud.

Required repository files:

app.py
churn_xgboost_model.pkl
churn_preprocessor.pkl
requirements.txt
README.md

The Gemini API key should be added through the deployment platform's secret-management system rather than committed to GitHub.

🧪 End-to-End Workflow
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Model Training
        ↓
Model Comparison
        ↓
Hyperparameter Optimization
        ↓
Final XGBoost Model
        ↓
Model + Preprocessor Saved
        ↓
Streamlit Application
        ↓
Churn Probability
        ↓
Risk Segmentation
        ↓
Gemini AI Explanation
        ↓
Retention Suggestions
⚠️ Important Note

The very high performance of the tree-based models should be interpreted carefully. The dataset shows strong feature separation, including a particularly strong relationship between contract type and churn. Therefore, these results may not represent performance on a more diverse real-world customer population.

For production use, the model should be validated on an independent real-world dataset and monitored after deployment.

🧰 Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Streamlit
Google Gemini API
RandomizedSearchCV
