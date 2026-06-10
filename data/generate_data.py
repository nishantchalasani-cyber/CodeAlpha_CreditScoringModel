import os
import numpy as np
import pandas as pd

def generate_credit_data(output_path: str, n_samples: int = 10000, seed: int = 42):
    """
    Generates a realistic credit scoring dataset resembling standard credit risk portfolios.
    Saves the result as a CSV file.
    """
    np.random.seed(seed)
    
    # 1. Age (person_age): Log-normal distribution centered around 27
    age = np.random.lognormal(mean=3.2, sigma=0.25, size=n_samples) + 15
    age = np.round(age).astype(int)
    # Clip and add some outliers
    age = np.clip(age, 18, 90)
    # Add a few unrealistic outliers to test cleanup (e.g., age > 100)
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.002), replace=False)
    age[outlier_indices] = np.random.randint(100, 145, size=len(outlier_indices))

    # 2. Annual Income (person_income): Log-normal distribution
    income = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples)
    income = np.round(income).astype(int)
    # Cap lower bound at 4000
    income = np.clip(income, 4000, 2000000)

    # 3. Employment Length (person_emp_length): Linked to age
    emp_length = []
    for a in age:
        max_emp = max(0, a - 16)
        if max_emp == 0:
            emp_length.append(0)
        else:
            # Most people have shorter employment, some long
            val = int(np.random.exponential(scale=5))
            emp_length.append(min(val, max_emp))
    emp_length = np.array(emp_length)

    # 4. Home Ownership (person_home_ownership)
    # Lower income -> Rent, Higher income -> Mortgage/Own
    home_ownership = []
    for inc in income:
        if inc < 35000:
            probs = [0.75, 0.15, 0.08, 0.02]  # Rent, Mortgage, Own, Other
        elif inc < 80000:
            probs = [0.45, 0.45, 0.09, 0.01]
        else:
            probs = [0.15, 0.70, 0.14, 0.01]
        home_ownership.append(np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], p=probs))
    home_ownership = np.array(home_ownership)

    # 5. Loan Intent (loan_intent)
    intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    loan_intent = np.random.choice(intents, size=n_samples, p=[0.20, 0.18, 0.18, 0.15, 0.14, 0.15])

    # 6. Loan Amount (loan_amnt)
    # Linked to income (people borrow relative to their income)
    loan_amnt = []
    for inc in income:
        # Borrow between 5% and 40% of income, centered around 15%
        ratio = np.random.beta(a=2, b=8) * 0.5 + 0.05
        amt = inc * ratio
        # Cap loan amounts between 500 and 35000
        amt = np.clip(amt, 500, 35000)
        loan_amnt.append(round(amt, -2))  # round to nearest hundred
    loan_amnt = np.array(loan_amnt)

    # 7. Interest Rate (loan_int_rate)
    # Corresponds to risk: lower for mortgage/own, higher for rent, higher for higher amounts
    loan_int_rate = []
    for i in range(n_samples):
        base_rate = 11.0
        if home_ownership[i] in ['OWN', 'MORTGAGE']:
            base_rate -= 2.0
        if income[i] > 100000:
            base_rate -= 1.5
        # Add loan size impact
        base_rate += (loan_amnt[i] / 10000) * 0.8
        rate = base_rate + np.random.normal(0, 2.5)
        loan_int_rate.append(round(np.clip(rate, 5.0, 23.5), 2))
    loan_int_rate = np.array(loan_int_rate)

    # 8. Credit History Length (cb_person_cred_hist_length): Linked to age
    cred_hist_len = []
    for a in age:
        if a < 22:
            hist = np.random.choice([2, 3, 4])
        else:
            # Let's say credit history started around age 18-22
            hist = int(a - np.random.randint(18, 23))
            hist = max(2, hist)
        # Handle outliers in age (clip history length)
        cred_hist_len.append(min(hist, 30 if a < 100 else 10))
    cred_hist_len = np.array(cred_hist_len)

    # 9. Previous Defaults (cb_person_default_on_file)
    # Higher chance if interest rate is high
    default_on_file = []
    for r in loan_int_rate:
        prob_def = 0.05 + (r - 5.0) * 0.015
        prob_def = np.clip(prob_def, 0.02, 0.6)
        default_on_file.append(np.random.choice(['Y', 'N'], p=[prob_def, 1.0 - prob_def]))
    default_on_file = np.array(default_on_file)

    # 10. Debt-to-Income Ratio (loan_percent_income)
    loan_percent_income = np.round(loan_amnt / income, 2)

    # 11. Target Variable: loan_status (Credit Risk)
    # Logistic default probability function
    # Let's construct logit weights
    logit = (
        -2.4 
        + 5.8 * loan_percent_income 
        + 1.8 * (default_on_file == 'Y') 
        + 0.9 * (home_ownership == 'RENT') 
        - 0.06 * emp_length 
        + 0.12 * (loan_int_rate - 11.0) 
        - 0.015 * (income / 10000.0)
        - 0.02 * cred_hist_len
    )
    
    # Calculate probability
    prob_default = 1 / (1 + np.exp(-logit))
    # Draw binary labels
    loan_status = np.random.binomial(1, prob_default)

    # Create DataFrame
    df = pd.DataFrame({
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length.astype(float),
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_status': loan_status,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': default_on_file,
        'cb_person_cred_hist_length': cred_hist_len
    })

    # Introduce missing values (realistic preprocessing test)
    # 3% missing in employment length
    emp_missing = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[emp_missing, 'person_emp_length'] = np.nan

    # 8% missing in interest rate
    rate_missing = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    df.loc[rate_missing, 'loan_int_rate'] = np.nan

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to {output_path} with {n_samples} records.")
    print(f"Class distribution:\n{df['loan_status'].value_counts(normalize=True)}")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_credit_data(os.path.join(out_dir, "credit_risk_dataset.csv"))
