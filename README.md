## Project Overview

Customer churn is a major business problem for subscription-based companies. The goal of this project is to help a business:

1. Predict which customers are likely to churn.
2. Estimate the probability of churn.
3. Segment customers into meaningful groups.
4. Identify high-priority customers for retention actions.
5. Expose the trained models through a FastAPI application.
6. Package the application with Docker for reproducible deployment.

The project follows an end-to-end Data Science workflow:

```text
Business Problem
      ↓
Data Validation
      ↓
EDA & Business Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train / Test Split
      ↓
Model Comparison
      ↓
Model Evaluation
      ↓
Threshold & Business Analysis
      ↓
Error Analysis
      ↓
Customer Segmentation
      ↓
Final Model Training
      ↓
FastAPI
      ↓
Docker
      ↓
Deployment-ready Application
```

---

## Business Problem

A telecom company wants to identify customers who are at risk of leaving the service.

A churn prediction system can help the business answer questions such as:

- Which customers are likely to churn?
- What is the probability that a customer will churn?
- Which customer groups require retention attention?
- Which customer characteristics are associated with higher churn risk?
- How can the prediction model be exposed as an application/API?

The objective is not only to build a model, but to connect the model with a practical business workflow.

---

## Dataset

The project uses the Telco Customer Churn dataset.

The raw dataset contains:

- **7,043 rows**
- **21 original columns**

Main fields include:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

The raw dataset is kept unchanged under:

```text
data/raw/Customer-Churn.csv
```

---

## Target Variable

The target variable is:

```text
Churn
```

It contains:

```text
No
Yes
```

For model training it is converted to:

```text
No  → 0
Yes → 1
```

Dataset target distribution:

```text
No Churn : 5174
Churn    : 1869
```

This means the dataset is imbalanced, so metrics such as Precision, Recall, F1-score, and ROC-AUC are considered instead of relying only on accuracy.

---

## Data Validation

Before modeling, the raw data is validated.

Validation includes:

- File existence
- Dataset shape
- Column inspection
- Data types
- Missing values
- Blank values
- Duplicate rows
- Customer ID uniqueness
- Numeric sanity checks
- Target validation
- Leakage screening
- `TotalCharges` investigation

The validation stage helps ensure that modeling is performed on trustworthy data.

---

## Exploratory Data Analysis

The EDA focuses on business-relevant churn patterns rather than generating unnecessary charts.

Important areas explored include:

- Overall churn distribution
- Customer tenure
- Tenure bands
- Contract type
- Monthly charges
- Total charges
- Internet service
- Support services
- Payment method
- Gender
- Partner status
- Dependents
- Senior citizen status
- Contract and tenure relationship
- Contract and monthly charge relationship

The objective of EDA is to understand customer behavior and identify patterns that can later support modeling and business decisions.

---

## Data Cleaning

`TotalCharges` is stored as a string in the raw dataset and contains blank-like values.

It is converted to numeric values.

For customers where `TotalCharges` is unavailable, the project reconstructs the value using:

```text
TotalCharges ≈ tenure × MonthlyCharges
```

The raw dataset is not modified.

---

## Feature Engineering

The project creates additional features to provide the model with more useful customer-level information.

### AverageMonthlySpend

```text
AverageMonthlySpend = TotalCharges / tenure
```

A lower bound of 1 is used for tenure during this calculation to avoid division by zero.

### IsNewCustomer

```text
IsNewCustomer = tenure <= 12
```

This identifies customers in their first year.

### ServiceCount

The project counts selected subscribed services:

- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

This creates a simple measure of service adoption.

---

## Model Preparation

The following steps are used:

- Separate customer identifier from modeling features.
- Convert target to binary.
- Split data into training and test sets.
- Use stratification for the target.
- Standardize numerical features.
- One-hot encode categorical features.
- Use `handle_unknown="ignore"` for categorical encoding.
- Fit preprocessing only on training data.

The train/test split uses:

```text
test_size = 0.20
random_state = 42
stratify = target
```

---

## Models Compared

The modeling stage compares several approaches:

### Baseline

```text
DummyClassifier
```

### Logistic Regression

```text
LogisticRegression
```

### Decision Tree

```text
DecisionTreeClassifier
```

### Random Forest

```text
RandomForestClassifier
```

### XGBoost

```text
XGBClassifier
```

### CatBoost

```text
CatBoostClassifier
```

The initial model comparison evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Final Churn Model

The final selected churn model is:

```text
Logistic Regression
```

The initial evaluation produced approximately:

```text
Accuracy : 0.80
ROC-AUC  : 0.84
```

For the churn class:

```text
Precision : 0.66
Recall    : 0.53
F1-score  : 0.59
```

Confusion matrix:

```text
True Negative  = 934
False Positive = 101
False Negative = 174
True Positive  = 200
```

The model provides churn probability rather than only a binary prediction.

---

## Business Threshold Analysis

The project also evaluates how changing the classification threshold affects business outcomes.

An illustrative cost framework is used:

```text
False Negative cost = ₹1000
False Positive cost = ₹100
```

At the default threshold of `0.50`, the illustrative cost is approximately:

```text
₹184,100
```

A business-oriented threshold analysis significantly reduces the illustrative cost to approximately:

```text
₹73,600
```

This demonstrates an important ML concept:

> The threshold that gives the best business outcome does not necessarily have to be the threshold that gives the best accuracy.

The production API currently uses:

```text
Churn threshold = 0.50
```

---

## Customer Segmentation

In addition to supervised churn prediction, the project performs unsupervised customer segmentation.

The clustering model does not use the churn target.

The final segmentation features are:

```text
tenure
MonthlyCharges
ServiceCount
```

The features are standardized before clustering.

K-Means clustering is evaluated across multiple values of `k`.

The final project uses:

```text
Number of clusters = 2
```

The segmentation artifacts are saved as:

```text
models/segmentation_scaler.joblib
models/customer_segmentation_kmeans.joblib
```

Customer-level segment assignments are saved to:

```text
data/processed/customer_segments.csv
```

Final segment distribution:

```text
Segment 0 : 3097 customers
Segment 1 : 3946 customers
```

The purpose of segmentation is to understand customer groups and support differentiated business actions.

---

## Project Structure

```text
Customer Churn Segmentation/
│
├── app/
│   ├── main.py
│   └── templates/
│       └── index.html
│
├── data/
│   ├── raw/
│   │   └── Customer-Churn.csv
│   └── processed/
│       └── customer_segments.csv
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── segmentation_scaler.joblib
│   └── customer_segmentation_kmeans.joblib
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_eda_business_analysis.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_customer_segmentation.ipynb
│
├── scripts/
│   └── train_final.py
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── segmentation/
│       └── clustering.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## FastAPI Application

The trained models are integrated into a FastAPI application.

Main endpoints:

### Home

```text
GET /
```

Returns the web application.

### Health Check

```text
GET /health
```

Used to verify that the application is running.

### Prediction

```text
POST /predict
```

Accepts customer information and returns:

- Churn probability
- Churn prediction
- Customer segment

### Swagger Documentation

When the application is running locally:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides interactive API documentation.

---

## Example API Response

A prediction returns information conceptually like:

```json
{
  "churn_probability": 0.72,
  "churn_prediction": 1,
  "segment": 0
}
```

Where:

```text
churn_probability → probability of churn
churn_prediction  → 0 = No Churn, 1 = Churn
segment           → K-Means customer segment
```

---

## Running Locally

### 1. Create/activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train final artifacts

```powershell
python .\scripts\train_final.py
```

This creates:

```text
models/churn_pipeline.joblib
models/segmentation_scaler.joblib
models/customer_segmentation_kmeans.joblib
data/processed/customer_segments.csv
```

### 4. Start FastAPI

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Testing

The project includes API tests using Pytest.

Run:

```powershell
pytest
```

Current test result:

```text
3 passed
```

The tests cover:

- Root endpoint
- Health endpoint
- Prediction endpoint

---

## Docker

The project is containerized using Docker.

### Build image

From the project root:

```powershell
docker build -t customer-churn-segmentation .
```

### Run container

```powershell
docker run -p 8000:8000 customer-churn-segmentation
```

Then open:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

The Docker container packages:

```text
Python
+
Dependencies
+
FastAPI application
+
Trained ML models
+
Application files
```

This makes the application easier to run in a consistent environment.

---

## Production / Deployment Direction

The current project is structured so that the application can later be deployed to a cloud platform.

The intended deployment architecture is:

```text
User
  ↓
Internet
  ↓
Cloud Server
  ↓
Docker Container
  ↓
FastAPI
  ↓
ML Models
  ↓
Prediction
```

Cloud deployment is intentionally treated as the next stage of the project rather than mixing deployment complexity into the core modeling workflow.

---

## Key Learnings

This project covers an end-to-end Data Science workflow:

### Data Science

- Data validation
- Exploratory Data Analysis
- Data cleaning
- Feature engineering
- Train/test splitting
- Preprocessing pipelines
- Classification
- Model comparison
- Model evaluation
- Threshold optimization
- Error analysis
- Permutation importance

### Machine Learning

- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- CatBoost
- Classification metrics
- ROC-AUC
- Precision-Recall trade-offs
- K-Means clustering

### Software Engineering

- Modular project structure
- Reusable preprocessing
- Model serialization
- API development
- Automated testing
- Configuration through project code
- Artifact validation

### Deployment

- FastAPI
- Docker
- Docker image creation
- Container execution
- Deployment-ready application architecture

---

## Why This Project Is More Than a Notebook

The project is designed as an end-to-end application rather than only a model-training notebook.

The workflow moves from:

```text
Raw Data
   ↓
Analysis
   ↓
Model
   ↓
Evaluation
   ↓
Saved Artifact
   ↓
API
   ↓
Docker
   ↓
Deployment
```

This makes the project useful for demonstrating both **Data Science** and **ML Engineering fundamentals**.

---

## Future Improvements

Potential future improvements include:

- Cloud deployment
- HTTPS
- Authentication
- Model monitoring
- Logging
- Data drift monitoring
- Experiment tracking
- Automated CI/CD
- More advanced customer segmentation
- Business dashboard
- Automated retraining pipeline

These are intentionally kept separate from the current core project to maintain a clear and understandable architecture.

---

## Tech Stack

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
CatBoost
Matplotlib
Seaborn
FastAPI
Pydantic
Pytest
Joblib
Docker
```

---

## Author

**Aman Soni**

Data Science / Machine Learning Portfolio Project

GitHub:

https://github.com/aman1165/customer-churn-segmentation

---

## Project Status

```text
Data Validation          ✅
EDA                      ✅
Data Preparation         ✅
Model Comparison         ✅
Model Evaluation         ✅
Final Training           ✅
Customer Segmentation    ✅
FastAPI                  ✅
Web UI                   ✅
Testing                  ✅
Docker                   🚧
Cloud Deployment         ⏳
```
"""

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(path)
