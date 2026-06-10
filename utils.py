import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Define features used in training
NUMERIC_FEATURES = [
    'person_age', 'person_income', 'person_emp_length', 
    'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income'
]
CATEGORICAL_FEATURES = ['person_home_ownership', 'loan_intent']
BINARY_FEATURES = ['cb_person_default_on_file']

class CreditDataPreprocessor:
    """
    Stateful preprocessor for Credit Risk data to handle missing values, outliers,
    encoding, and scaling without data leakage.
    """
    def __init__(self):
        self.median_values = {}
        self.one_hot_encoder = None
        self.scaler = None
        self.feature_names_out_ = None
        self.categories_ = {}

    def fit(self, df: pd.DataFrame):
        # 1. Calculate medians for numerical features on clean data
        clean_df = df.copy()
        
        # Outlier handling (pre-imputation and scaling)
        clean_df['person_age'] = np.clip(clean_df['person_age'], 18, 100)
        clean_df['person_emp_length'] = np.clip(clean_df['person_emp_length'], 0, 60)
        
        # Compute medians
        for col in NUMERIC_FEATURES:
            self.median_values[col] = float(clean_df[col].median(skipna=True))
            
        # 2. Fit OneHotEncoder
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.one_hot_encoder.fit(clean_df[CATEGORICAL_FEATURES].fillna('OTHER'))
        
        # 3. Fit StandardScaler on scaled features
        # To scale, we impute missing and encode first
        encoded_cats = self.one_hot_encoder.transform(clean_df[CATEGORICAL_FEATURES].fillna('OTHER'))
        cat_feature_names = self.one_hot_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        
        # Binary column mapping: Y -> 1, N -> 0
        binary_mapped = clean_df[BINARY_FEATURES].copy()
        for col in BINARY_FEATURES:
            binary_mapped[col] = binary_mapped[col].map({'Y': 1, 'N': 0}).fillna(0).astype(float)
            
        # Impute numericals
        imputed_nums = clean_df[NUMERIC_FEATURES].copy()
        for col in NUMERIC_FEATURES:
            imputed_nums[col] = imputed_nums[col].fillna(self.median_values[col])
            
        # Merge all numerical features for scaling
        all_features = np.hstack([imputed_nums.values, binary_mapped.values, encoded_cats])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)
        
        # Save output feature names
        self.feature_names_out_ = NUMERIC_FEATURES + BINARY_FEATURES + list(cat_feature_names)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_clean = df.copy()
        
        # 1. Clean outliers
        df_clean['person_age'] = np.clip(df_clean['person_age'], 18, 100)
        df_clean['person_emp_length'] = np.clip(df_clean['person_emp_length'], 0, 60)
        
        # Prevent employment length from exceeding age - 16
        df_clean['person_emp_length'] = np.minimum(
            df_clean['person_emp_length'], 
            np.maximum(0, df_clean['person_age'] - 16)
        )
        
        # 2. Impute missing values
        for col in NUMERIC_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(self.median_values[col])
                
        # 3. Process categorical columns
        cat_data = df_clean[CATEGORICAL_FEATURES].fillna('OTHER')
        encoded_cats = self.one_hot_encoder.transform(cat_data)
        
        # 4. Binary mapping
        binary_mapped = df_clean[BINARY_FEATURES].copy()
        for col in BINARY_FEATURES:
            binary_mapped[col] = binary_mapped[col].map({'Y': 1, 'N': 0}).fillna(0).astype(float)
            
        # 5. Combine and Scale
        imputed_nums = df_clean[NUMERIC_FEATURES].values
        merged_features = np.hstack([imputed_nums, binary_mapped.values, encoded_cats])
        
        return self.scaler.transform(merged_features)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

# PLOTTING UTILITIES
def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x='loan_status', data=df, palette=['#1f77b4', '#d62728'])
    plt.title('Credit Risk Class Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Credit Risk (0 = Good, 1 = Bad)', fontsize=12)
    plt.ylabel('Count of Applicants', fontsize=12)
    ax.set_xticklabels(['Good Credit Risk', 'Bad Credit Risk'])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_income_distribution(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(8, 4.5))
    sns.set_theme(style="darkgrid")
    # Filter income outlier for visualization readability
    filtered_df = df[df['person_income'] < 200000]
    sns.histplot(data=filtered_df, x='person_income', hue='loan_status', multiple='stack', 
                 palette=['#2ca02c', '#d62728'], bins=40, kde=True)
    plt.title('Annual Income Distribution by Loan Status', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Annual Income (₹)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_loan_amount_distribution(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(8, 4.5))
    sns.set_theme(style="darkgrid")
    sns.kdeplot(data=df, x='loan_amnt', hue='loan_status', fill=True, 
                palette=['#2ca02c', '#d62728'], common_norm=False, alpha=0.5)
    plt.title('Loan Amount Density by Loan Status', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Loan Amount (₹)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    # Keep numerical values and map binary
    temp_df = df.copy()
    temp_df['cb_person_default_on_file'] = temp_df['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).fillna(0)
    
    # Select columns of interest
    cols = NUMERIC_FEATURES + ['cb_person_default_on_file', 'loan_status']
    corr_matrix = temp_df[cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of Key Credit Variables', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_feature_importance(importances: np.ndarray, feature_names: list, save_path: str = None):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices[:15]]
    sorted_importances = importances[indices[:15]]
    
    sns.barplot(x=sorted_importances, y=sorted_names, palette="viridis")
    plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Relative Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="darkgrid")
    melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='muted')
    plt.title('Model Performance Benchmark Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Machine Learning Models', fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def generate_pdf_report(client_data: dict, risk_results: dict, file_path: str):
    """
    Generates a professional fintech credit assessment report.
    """
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    doc = SimpleDocTemplate(
        file_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=30
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    bold_body = ParagraphStyle(
        'BoldBodyCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Score details
    score_style = ParagraphStyle(
        'ScoreVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=colors.HexColor('#0F172A'),
        alignment=1 # Center
    )
    
    status_style = ParagraphStyle(
        'StatusVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        alignment=1 # Center
    )

    story = []
    
    # 1. Header
    story.append(Paragraph("CREDIT ELIGIBILITY ASSESSMENT REPORT", title_style))
    story.append(Paragraph("CodeAlpha Loan Risk Underwriting Engine &bull; Automated System Audit", subtitle_style))
    
    # Divider line
    divider = Table([[""]], colWidths=[530], rowHeights=[3])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#3B82F6')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 20))
    
    # 2. Results Block (Status & Score)
    status = risk_results['status']
    score = risk_results['score']
    prob = risk_results['probability']
    recommendation = risk_results['recommendation']
    
    # Color coding status
    if status in [
    "Excellent Credit",
    "Good Credit",
    "Fair Credit"
        ]:
        status_color = colors.HexColor('#10B981')  # Green
        status_text = "APPROVED (LOW RISK)"
    else:
        status_color = colors.HexColor('#EF4444')  # Red
        status_text = "REJECTED (HIGH RISK)"
        
    status_p = Paragraph(status_text, ParagraphStyle('St', parent=status_style, textColor=status_color))
    score_p = Paragraph(f"{score} / 900", score_style)
    prob_p = Paragraph(f"Risk Probability: {prob:.2f}%", ParagraphStyle('Pr', parent=body_style, alignment=1))
    
    metric_table_data = [
        [Paragraph("Decision Status", ParagraphStyle('H', parent=bold_body, alignment=1)), 
         Paragraph("Credit Risk Score", ParagraphStyle('H2', parent=bold_body, alignment=1))],
        [status_p, score_p],
        ["", prob_p]
    ]
    
    metric_table = Table(metric_table_data, colWidths=[265, 265])
    metric_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-2), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    
    story.append(metric_table)
    story.append(Spacer(1, 20))
    
    # 3. Applicant Information Section
    story.append(Paragraph("Applicant & Loan Details", section_title))
    
    # Construct applicant table
    app_details = [
        [Paragraph("Age", bold_body), Paragraph(str(client_data['person_age']), body_style),
         Paragraph("Annual Income", bold_body), Paragraph(f"Rs. {client_data['person_income']:,}", body_style)],
        
        [Paragraph("Employment Length", bold_body), Paragraph(f"{int(client_data['person_emp_length'])} Years", body_style),
         Paragraph("Home Ownership", bold_body), Paragraph(client_data['person_home_ownership'], body_style)],
        
        [Paragraph("Loan Intent", bold_body), Paragraph(client_data['loan_intent'], body_style),
         Paragraph("Loan Amount", bold_body), Paragraph(f"Rs. {client_data['loan_amnt']:,}", body_style)],
        
        [Paragraph("Interest Rate", bold_body), Paragraph(f"{client_data['loan_int_rate']}%", body_style),
         Paragraph("Credit Hist. Length", bold_body), Paragraph(f"{int(client_data['cb_person_cred_hist_length'])} Years", body_style)],
        
        [Paragraph("Previous Default", bold_body), Paragraph(client_data['cb_person_default_on_file'], body_style),
         Paragraph("Debt-to-Income (DTI)", bold_body), Paragraph(f"{client_data['loan_percent_income'] * 100:.1f}%", body_style)]
    ]
    
    app_table = Table(app_details, colWidths=[130, 135, 130, 135])
    app_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#F8FAFC')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    story.append(app_table)
    story.append(Spacer(1, 20))
    
    # 4. Detailed Recommendation
    story.append(Paragraph("System Recommendation Details", section_title))
    story.append(Paragraph(recommendation, body_style))
    story.append(Spacer(1, 40))
    
    # 5. Signatures and Disclaimers
    signature_data = [
        [Paragraph("<b>Audit System Signature</b>", body_style), Paragraph("<b>Credit Committee Approval</b>", body_style)],
        ["", ""],  # Spacer row
        [Paragraph("_____________________________<br/>Auto-Underwriting System API", subtitle_style),
         Paragraph("_____________________________<br/>Authorized Underwriting Officer", subtitle_style)]
    ]
    sig_table = Table(signature_data, colWidths=[265, 265])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('BOTTOMPADDING', (0,1), (-1,1), 30), # Space for signing
    ]))
    
    story.append(sig_table)
    
    doc.build(story)
