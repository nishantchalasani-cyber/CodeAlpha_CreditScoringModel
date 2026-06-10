import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Import custom preprocessing and plotting utilities
from utils import (
    CreditDataPreprocessor, 
    plot_class_distribution, 
    plot_income_distribution, 
    plot_loan_amount_distribution, 
    plot_correlation_heatmap, 
    plot_feature_importance, 
    plot_model_comparison
)

def train_and_evaluate():
    # 1. Load Data
    data_path = os.path.join("data", "credit_risk_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Generate and save exploratory charts
    print("Generating EDA charts...")
    os.makedirs("model", exist_ok=True)
    plot_class_distribution(df, save_path=os.path.join("model", "class_distribution.png"))
    plot_income_distribution(df, save_path=os.path.join("model", "income_distribution.png"))
    plot_loan_amount_distribution(df, save_path=os.path.join("model", "loan_amount_distribution.png"))
    plot_correlation_heatmap(df, save_path=os.path.join("model", "correlation_heatmap.png"))

    # Split into features (X) and target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")

    # 2. Fit Preprocessor
    print("Fitting preprocessor...")
    preprocessor = CreditDataPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    # Save Scaler and Preprocessor
    joblib.dump(preprocessor, os.path.join("model", "preprocessor.pkl"))
    joblib.dump(preprocessor.scaler, os.path.join("model", "scaler.pkl"))
    print("Preprocessor and scaler saved.")

    # 3. Train Models
    models = {
        'Logistic Regression': LogisticRegression(
                                max_iter=1000,
                                 random_state=42,
                                 class_weight='balanced'
                                ),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
            ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
             max_depth=6,
             learning_rate=0.05,
             subsample=0.8,
             colsample_bytree=0.8,
              random_state=42,
             eval_metric='logloss'
                )
    }
    results = {}
    trained_models = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_clean, y_train)
        
        # Predict
        y_pred = model.predict(X_test_clean)
        y_prob = model.predict_proba(X_test_clean)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization

        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1 Score': float(f1),
            'ROC-AUC': float(roc),
            'Confusion Matrix': cm
        }
        trained_models[name] = model
        print(f"{name} Metrics: F1={f1:.4f}, Accuracy={acc:.4f}, ROC-AUC={roc:.4f}")

    # Save Comparison Chart
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1 Score'],
            'ROC-AUC': metrics['ROC-AUC']
        })
    comparison_df = pd.DataFrame(comparison_data)
    plot_model_comparison(comparison_df, save_path=os.path.join("model", "model_comparison.png"))

    # 4. Auto-Select Best Model
    # We select based on F1-score to handle class imbalance
    best_model_name = max(results, key=lambda k: results[k]['F1 Score'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n>>> Best model selected: {best_model_name} with F1-score: {best_metrics['F1 Score']:.4f}")
    print("\nBest Model Metrics")
    print(best_metrics)

    # Save Best Model Info
    model_metadata = {
        'best_model_name': best_model_name,
        'metrics': results,
        'feature_names': preprocessor.feature_names_out_
    }
    
    with open(os.path.join("model", "metrics.json"), "w") as f:
        json.dump(model_metadata, f, indent=4)
        
    # Save the best model
    joblib.dump(best_model, os.path.join("model", "best_model.pkl"))
    # Also save to root as credit_model.pkl as required by project structure
    joblib.dump(best_model, "credit_model.pkl")
    print(f"Saved best model to model/best_model.pkl and credit_model.pkl")

    # 5. Save Feature Importance and Confusion Matrix for the Best Model
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(preprocessor.feature_names_out_))
        
    plot_feature_importance(
        importances, 
        preprocessor.feature_names_out_, 
        save_path=os.path.join("model", "feature_importance.png")
    )
    
    # Save best model confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(best_metrics['Confusion Matrix'])
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Good Risk', 'Bad Risk'], yticklabels=['Good Risk', 'Bad Risk'])
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join("model", "confusion_matrix.png"), dpi=300)
    plt.close()
    print("=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best Model: {best_model_name}")
    print("=" * 50)

if __name__ == "__main__":
    train_and_evaluate()
