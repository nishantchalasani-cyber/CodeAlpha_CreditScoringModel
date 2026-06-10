# 💳 Credit Scoring System

A Machine Learning-based Credit Scoring System built using Python, Scikit-Learn, XGBoost, and Streamlit.

This project was developed as part of the **CodeAlpha Machine Learning Internship**. It simulates how banks and financial institutions evaluate loan applications by analyzing applicant information, estimating default risk, incorporating CIBIL scores, and generating credit decisions.

The application provides an interactive underwriting dashboard, calculates a credit score, estimates loan eligibility, and generates downloadable PDF assessment reports.

---

## 📌 Project Overview

Credit scoring plays a critical role in modern banking and lending. Before approving a loan, financial institutions need to assess whether an applicant is likely to repay the loan or default in the future.

This project automates that process using Machine Learning and a hybrid credit evaluation approach.

The system:

- Analyzes applicant financial information
- Predicts probability of default
- Incorporates applicant CIBIL score
- Calculates a final credit score
- Classifies risk level
- Estimates eligible loan amount
- Generates professional PDF reports

---

## 🚀 Key Features

### ✅ Data Preprocessing Pipeline

- Missing value handling
- Outlier treatment
- One-Hot Encoding for categorical variables
- Feature scaling using StandardScaler
- Prevention of data leakage

---

### ✅ Machine Learning Model Benchmarking

The project trains and compares multiple models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best-performing model is automatically selected using **F1 Score**, which is better suited for imbalanced credit-risk datasets.

---

### ✅ Hybrid Credit Scoring System

The final credit score combines:

- Machine Learning Risk Score
- Applicant CIBIL Score

This creates a more realistic credit evaluation process.

#### Credit Score Formula

```text
Final Credit Score =
(50% × ML Risk Score)
+
(50% × Applicant CIBIL Score)
```

---

### ✅ Credit Score Categories

| Score Range | Category |
|------------|----------|
| 750 – 900 | Excellent Credit |
| 700 – 749 | Good Credit |
| 650 – 699 | Fair Credit |
| 600 – 649 | Poor Credit |
| Below 600 | Very Poor Credit |

---

### ✅ Loan Eligibility Estimation

The system estimates eligible loan amounts based on:

- Income
- Credit Score
- Risk Category

This simulates a simplified underwriting decision process.

---

### ✅ Interactive Streamlit Dashboard

#### Dashboard

Displays:

- Total applicants
- Average annual income
- Average loan amount
- Historical default rate
- Model performance comparison

#### Credit Prediction

Allows users to:

- Enter applicant details
- Input CIBIL score
- Generate calculated credit score
- View risk category
- Estimate loan eligibility
- Download PDF reports

#### Analytics

Provides:

- Age distribution analysis
- Debt-to-Income analysis
- Correlation heatmaps
- Feature importance visualizations

---

### ✅ PDF Assessment Reports

The system automatically generates professional PDF reports containing:

- Applicant profile
- Loan details
- Credit score
- Risk category
- Approval / Rejection decision
- Underwriting recommendation

---

## 🧠 How the System Works

### Step 1

The Machine Learning model evaluates:

- Applicant age
- Income
- Employment history
- Loan amount
- Interest rate
- Credit history length
- Previous defaults
- Home ownership status
- Debt-to-Income ratio

### Step 2

The model predicts the probability that an applicant may default on a loan.

### Step 3

The default probability is converted into an ML Risk Score ranging from **300 to 900**.

### Step 4

The ML Risk Score is combined with the applicant's CIBIL score to generate the final credit score.

### Step 5

The system classifies the applicant into a risk category and provides an underwriting recommendation.

---

## 📂 Dataset Information

The dataset contains approximately **10,000 loan applicants**.

| Feature | Description |
|----------|-------------|
| person_age | Applicant age |
| person_income | Annual income (₹) |
| person_home_ownership | Home ownership status |
| person_emp_length | Employment length (Years) |
| loan_intent | Loan purpose |
| loan_amnt | Requested loan amount (₹) |
| loan_int_rate | Loan interest rate (%) |
| cb_person_default_on_file | Previous default history |
| cb_person_cred_hist_length | Credit history length |
| loan_percent_income | Debt-to-Income Ratio |
| loan_status | Target Variable (0 = Good Risk, 1 = Bad Risk) |

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/CodeAlpha_CreditScoringModel.git

cd CodeAlpha_CreditScoringModel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models

```bash
py train.py
```

### 4. Launch Application

```bash
py -m streamlit run app.py
```

---

## 🏆 Model Performance

Models were evaluated using an 80/20 train-test split.

Since credit-risk datasets are naturally imbalanced, **F1 Score** was selected as the primary evaluation metric.

| Model | Accuracy | F1 Score | ROC-AUC |
|---------|---------:|---------:|---------:|
| Logistic Regression | 71.75% | 48.96% | 78.22% |
| Decision Tree | 75.90% | 47.61% | 74.36% |
| Random Forest | 81.75% | 36.74% | 76.92% |
| XGBoost | 80.95% | 38.65% | 75.32% |

### Best Model Selected

**Logistic Regression**

Selected automatically because it achieved the highest F1 Score on the test dataset.

---

## 📸 Application Screenshots

### Dashboard

Portfolio statistics and model benchmarking.

### Credit Prediction

Applicant underwriting simulator with calculated credit score and loan eligibility estimation.

### Analytics

Feature importance analysis, DTI analysis, and correlation visualizations.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- ReportLab

---

## 🔮 Future Improvements

Potential future enhancements include:

- Hyperparameter tuning using GridSearchCV and Optuna
- Cross-validation pipelines
- Explainable AI using SHAP
- Real Credit Bureau API integration
- EMI Calculator
- FastAPI backend integration
- Cloud deployment using AWS or Streamlit Cloud

---

## 👨‍💻 Author

Developed as part of the **CodeAlpha Machine Learning Internship Program**.

---

## 📜 Disclaimer

This project was developed for educational and portfolio purposes only.

It is not intended for real-world lending decisions, banking operations, or financial advisory services.