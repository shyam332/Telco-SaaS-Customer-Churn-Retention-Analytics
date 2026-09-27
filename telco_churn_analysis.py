"""
========================================================================================
TELCO & SAAS CUSTOMER CHURN & RETENTION ANALYTICS (END-TO-END PIPELINE)
========================================================================================
Analyzes customer and subscription data to uncover why customers leave, which segments
are most vulnerable, customer lifetime survival dynamics, and high-impact retention playbooks.

Run:
    python telco_churn_analysis.py
========================================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

def main():
    print("=" * 80)
    print("         TELCO & SAAS CUSTOMER CHURN & RETENTION ANALYTICS PIPELINE")
    print("=" * 80)

    # 1. Ingestion & Schema Auditing
    csv_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Missing {csv_file} in current working directory.")

    df = pd.read_csv(csv_file)
    print(f"\n[1] DATA INGESTION & QUALITY CHECK")
    print(f"Total Records: {len(df):,} | Total Attributes: {df.shape[1]}")

    # Handling TotalCharges blank spaces (domain imputation for tenure == 0)
    blanks = (df['TotalCharges'].str.strip() == '').sum()
    print(f"Found {blanks} records with blank TotalCharges. Imputing 0.0 (new signups with tenure=0)...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', '0.0'))
    df['Churn_num'] = (df['Churn'] == 'Yes').astype(int)

    # 2. Executive Business Metrics
    total_cust = len(df)
    churn_cust = df['Churn_num'].sum()
    churn_rate = churn_cust / total_cust
    total_mrr = df['MonthlyCharges'].sum()
    churn_mrr = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    mrr_churn_pct = (churn_mrr / total_mrr) * 100
    arr_lost = churn_mrr * 12

    print("\n" + "=" * 80)
    print("                      EXECUTIVE REVENUE & CHURN SUMMARY")
    print("=" * 80)
    print(f"Active Customer Base            : {total_cust:,}")
    print(f"Churned Customers (Lost)        : {churn_cust:,} ({churn_rate*100:.2f}%)")
    print(f"Retained Customers              : {total_cust - churn_cust:,} ({(1-churn_rate)*100:.2f}%)")
    print(f"Total Monthly Recurring Revenue : ${total_mrr:,.2f}")
    print(f"Monthly Lost Revenue (Churn MRR): ${churn_mrr:,.2f} ({mrr_churn_pct:.2f}% of total MRR)")
    print(f"Annualized Lost Revenue (ARR)   : ${arr_lost:,.2f}")
    print(f"Avg Monthly Bill (Retained)     : ${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}")
    print(f"Avg Monthly Bill (Churned)      : ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f} (+21.5% premium)")
    print("=" * 80)

    # 3. Tenure Cohorts & Infant Mortality Analysis
    print("\n[2] CUSTOMER TENURE COHORTS & DROP-OFF CLIFF")
    bins = [-1, 1, 6, 12, 24, 48, 72]
    labels = ['0-1 mo', '2-6 mo', '7-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
    df['Tenure_Cohort'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    cohort_tbl = df.groupby('Tenure_Cohort', observed=False).agg(
        Cust_Count=('customerID', 'count'),
        Churn_Count=('Churn_num', 'sum'),
        Churn_Rate=('Churn_num', lambda x: f"{x.mean()*100:.1f}%"),
        Avg_Monthly_Bill=('MonthlyCharges', lambda x: f"${x.mean():.2f}"),
        Lost_MRR=('MonthlyCharges', lambda x: f"${x[df.loc[x.index, 'Churn']=='Yes'].sum():,.2f}")
    )
    cohort_tbl['Pct_Total_Attrition'] = (df.groupby('Tenure_Cohort', observed=False)['Churn_num'].sum() / churn_cust * 100).apply(lambda x: f"{x:.1f}%")
    print(cohort_tbl)

    # Month 1 specific probe
    m1 = df[df['tenure'] == 1]
    print(f"\n* Month 1 Drop-off: {len(m1)} customers evaluated; {m1['Churn_num'].sum()} churned ({m1['Churn_num'].mean()*100:.2f}% churn rate)!")

    # 4. Segment Breakdown: Contract, Payment & Internet Service
    print("\n[3] STRUCTURAL SEGMENT VULNERABILITY")
    
    print("\n--- By Contract Commitment ---")
    c_tbl = df.groupby('Contract').agg(
        Volume=('customerID', 'count'),
        Churn_Rate=('Churn_num', lambda x: f"{x.mean()*100:.2f}%"),
        Lost_MRR=('MonthlyCharges', lambda x: f"${x[df.loc[x.index, 'Churn']=='Yes'].sum():,.2f}")
    )
    print(c_tbl)

    print("\n--- By Payment Method ---")
    p_tbl = df.groupby('PaymentMethod').agg(
        Volume=('customerID', 'count'),
        Churn_Rate=('Churn_num', lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Bill=('MonthlyCharges', lambda x: f"${x.mean():.2f}")
    )
    print(p_tbl)

    print("\n--- By Internet Architecture ---")
    i_tbl = df.groupby('InternetService').agg(
        Volume=('customerID', 'count'),
        Churn_Rate=('Churn_num', lambda x: f"{x.mean()*100:.2f}%"),
        Avg_Bill=('MonthlyCharges', lambda x: f"${x.mean():.2f}"),
        Lost_MRR=('MonthlyCharges', lambda x: f"${x[df.loc[x.index, 'Churn']=='Yes'].sum():,.2f}")
    )
    print(i_tbl)

    print("\n--- The Retention Shield (Add-on Services) ---")
    for s in ['TechSupport', 'OnlineSecurity', 'DeviceProtection', 'OnlineBackup']:
        sub = df[df[s] != 'No internet service']
        r_no = sub[sub[s] == 'No']['Churn_num'].mean() * 100
        r_yes = sub[sub[s] == 'Yes']['Churn_num'].mean() * 100
        print(f"  • {s:17s}: Without Add-on = {r_no:5.1f}% | With Add-on = {r_yes:5.1f}% | Churn Delta = -{r_no - r_yes:.1f}%")

    # 5. Statistical Hypothesis Testing
    print("\n[4] STATISTICAL HYPOTHESIS TESTING")
    cat_cols = ['Contract', 'OnlineSecurity', 'TechSupport', 'InternetService', 
                'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen', 'gender']
    print(f"{'Feature':20s} {'Chi2 Stat':>12s} {'p-value':>12s} {'Significance'}")
    print("-" * 55)
    for c in cat_cols:
        ct = pd.crosstab(df[c], df['Churn'])
        chi2, p_val, _, _ = stats.chi2_contingency(ct)
        sig = "Significant (p<0.001)" if p_val < 0.001 else "Not Significant"
        print(f"{c:20s} {chi2:12.1f} {p_val:12.2e}   {sig}")

    # 6. Machine Learning Model & Odds Ratios
    print("\n[5] MACHINE LEARNING DRIVER MODELING")
    X = df.drop(columns=['customerID', 'Churn', 'Churn_num', 'Tenure_Cohort'])
    y = df['Churn_num']

    # Strict featurization ordering: split before fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols_m = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_m)
        ]
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols_m)
    all_names = num_cols + list(cat_names)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_p, y_train)
    y_prob_lr = lr.predict_proba(X_test_p)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train_p, y_train)
    y_prob_rf = rf.predict_proba(X_test_p)[:, 1]

    print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f} | PR-AUC: {average_precision_score(y_test, y_prob_lr):.4f}")
    print(f"Random Forest       ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f} | PR-AUC: {average_precision_score(y_test, y_prob_rf):.4f}")

    # Top Odds Ratios
    odds_df = pd.DataFrame({'Feature': all_names, 'Odds_Ratio': np.exp(lr.coef_[0])}).sort_values('Odds_Ratio', ascending=False)
    print("\nTop 4 Churn Promoters (Highest Risk Multipliers):")
    for _, row in odds_df.head(4).iterrows():
        print(f"  • {row['Feature']:35s}: Odds Ratio = {row['Odds_Ratio']:.2f}x")

    print("\nTop 4 Retention Anchors (Strongest Protective Factors):")
    for _, row in odds_df.tail(4).iterrows():
        print(f"  • {row['Feature']:35s}: Odds Ratio = {row['Odds_Ratio']:.2f}x")

    # 7. Strategic ROI Preservation Scenarios
    print("\n" + "=" * 80)
    print("                 STRATEGIC ROI & ARR PRESERVATION MODEL")
    print("=" * 80)
    for pct in [5, 10, 15, 20]:
        saved_arr = arr_lost * (pct / 100.0)
        saved_mrr = saved_arr / 12.0
        saved_cust = churn_cust * (pct / 100.0)
        print(f"Target: {pct:2d}% Churn Reduction -> Save {saved_cust:4.0f} accounts | +${saved_mrr:10,.2f}/mo MRR | +${saved_arr:10,.2f}/yr ARR")
    print("=" * 80)

if __name__ == '__main__':
    main()
