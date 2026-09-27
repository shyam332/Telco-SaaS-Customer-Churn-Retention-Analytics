# 📊 Telco & SaaS Customer Churn & Retention Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20%7C%20IBM%20Telco-orange.svg)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

An end-to-end, production-grade analytics study examining customer subscription dynamics, tenure survival curves, cohort drop-offs, and high-impact retention playbooks for SaaS and subscription businesses.

Grounding dataset: **7,043 customer accounts** | **$456,117 Baseline Monthly Recurring Revenue (MRR)**.

---

## 🎯 Executive Overview & Core Questions Answered

In subscription businesses, reducing customer churn directly drives Net Revenue Retention (NRR), shortens Customer Acquisition Cost (CAC) payback periods, and expands Customer Lifetime Value (LTV).

This study addresses the four foundational questions faced by Product, Growth, and Retention teams:

| Business Question | Data-Backed Finding | Strategic Intervention |
| :--- | :--- | :--- |
| **1. Why are customers leaving?** | **Early onboarding failure** (62.0% exit in Month 1), **price-to-value friction** on premium fiber optic (\$91.50/mo) lacking tech support, and **friction-heavy billing rails** (Electronic Check). | Fix onboarding friction in the first 90 days and bundle technical support into high-tier subscriptions. |
| **2. Which segments are most likely to churn?** | **Month-to-Month contracts** (42.7% churn, **86.9% of lost MRR**), **Fiber Optic without Tech Support** (57.5% churn), **Electronic Check payers** (45.3% churn), and **Senior Citizens** (41.7% churn). | Target Month-to-Month and Fiber subscribers with high-leverage switching incentives and support bundles. |
| **3. How long do customers stay active?** | Median active lifespan for churned users is only **10 months** (vs. 38 months for retained). **55.5% of all churn occurs in Year 1**, featuring an acute **Month 1 drop-off cliff (62.0% churn)**. | Shift retention resources to Days 1–90 rather than waiting until renewal at Month 12. |
| **4. What actions improve retention?** | Four SaaS Retention Playbooks: **First 90 Days Care**, **Annual Contract Migration**, **Fiber Total Care**, and **Auto-Pay \$5 Incentives**. | Reclaiming just 10% of churned accounts preserves **\$166,957 in ARR** annually; a 15% reduction recovers **\$250,436 in ARR**. |

---

## 💡 Key Business & Revenue Findings

1. **Revenue Asymmetry**: 
   - Logo churn is **26.54%** (1,869 / 7,043 customers).
   - Monthly Recurring Revenue (MRR) churn is **30.50%** (**\$139,131 / month**, or **\$1.67 Million ARR**).
   - Churned customers pay an average monthly bill of **\$74.44**, compared to **\$61.27** for retained customers (**+21.5% premium**). The business is disproportionately bleeding high-value accounts.
2. **The Month 1 "Infant Mortality" Cliff**:
   - Out of 613 customers evaluated at Month 1, **380 churned immediately (61.99% churn rate)**.
   - Over **20% of all platform attrition occurs in Month 1**, and **55.5% occurs within the first 12 months**.
3. **The "Fiber Optic Paradox" & The Add-on Shield**:
   - Fiber Optic subscribers exhibit a **41.89% churn rate** despite having high-speed infrastructure, generating **\$114,300/mo of lost MRR (82.2% of all lost revenue)**.
   - Adding **Tech Support** reduces churn from 41.6% down to **15.2%** (-26.4% delta).
   - Adding **Online Security** reduces churn from 41.8% down to **14.6%** (-27.2% delta).
4. **Contract & Payment Friction**:
   - Month-to-Month contracts churn at **42.71%**, vs. **11.27%** for 1-Year plans and **2.83%** for 2-Year plans (over **15x lower churn**).
   - Manual Electronic Check payers churn at **45.29%**, compared to **15.24%** for automatic Credit Card payments.

---

## 📁 Repository Structure

```text
telco-customer-churn-analysis/
├── README.md                                     # Project overview and executive summary
├── requirements.txt                              # Frozen Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Base dataset (7,043 rows, 21 columns)
├── telco_churn_analysis.py                       # End-to-end standalone Python analytics pipeline
├── generate_charts.py                            # Publication-quality chart generator
├── build_notebook.py                             # Script that programmatically builds the Jupyter Notebook
├── Telco_Customer_Churn_Retention_Analysis.ipynb # Reproducible notebook following ML best practices
└── charts/                                       # High-resolution analytical visualizations (300 DPI)
    ├── churn_kpi_revenue.png                     # Logo vs. MRR churn decomposition
    ├── survival_curves.png                       # Kaplan-Meier survival curves (overall & by contract)
    ├── contract_payment_friction.png             # Contract commitment & payment rail friction
    ├── service_ecosystem_shield.png              # Impact of technical support & security add-ons
    ├── pricing_elasticity_shock.png              # Density distribution of monthly charges & tiers
    ├── ml_drivers_importance.png                 # Logistic regression odds ratios & RF importance
    └── cohort_retention_heatmap.png              # Contract vs. tenure cohort churn matrix
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:

```bash
cd telco-customer-churn-analysis
```

Initialize a clean virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Analytics Pipeline
Execute the complete end-to-end analysis (data hygiene, metrics, hypothesis tests, ML models, and ROI tables):

```bash
python telco_churn_analysis.py
```

### 3. Generate Analytical Visualizations
Generate all 7 publication-ready figures in the `charts/` folder:

```bash
python generate_charts.py
```

### 4. Interactive Jupyter Notebook
Launch Jupyter or open the notebook in VS Code:

```bash
jupyter notebook Telco_Customer_Churn_Retention_Analysis.ipynb
```

---

## 🔬 Methodology & Analytical Rigor

### 1. Data Hygiene & Missing Value Resolution
- **Anomalous Records**: 11 customer records had blank string values (`' '`) in `TotalCharges`.
- **Root Cause**: All 11 records have `tenure == 0` (brand-new accounts signed up in the current billing cycle). None of them have churned (`Churn == 'No'`).
- **Handling**: Rather than dropping valid customer records, `TotalCharges` was converted to `float64` and imputed with `0.0`.

### 2. Survival Analysis (Kaplan-Meier Estimation)
Non-parametric survival functions $S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$ were computed across 72 monthly tenure intervals to isolate hazard milestones:
- Month 1: $S(1) = 94.6\%$ (380 exits)
- Month 12: $S(12) = 84.3\%$
- Month 24: $S(24) = 78.9\%$
- Month 48: $S(48) = 70.9\%$

### 3. Statistical Hypothesis Testing
- **Categorical Associations**: Chi-Square ($\chi^2$) tests of independence confirmed that `Contract` ($\chi^2 = 1,184.6, p < 10^{-250}$), `OnlineSecurity` ($\chi^2 = 850.0$), and `TechSupport` ($\chi^2 = 828.2$) are the primary drivers of churn. `gender` ($p = 0.487$) and `PhoneService` ($p = 0.339$) were proven statistically insignificant.
- **Continuous Differences**: Mann-Whitney U tests verified significant rank-sum differences between retained and churned cohorts for `tenure` ($p < 10^{-200}$) and `MonthlyCharges` ($p < 10^{-50}$).

### 4. Supervised Machine Learning & Interpretability
- **Strict Featurization Ordering**: Stratified 80/20 train/test split was performed **before** fitting scalers and encoders to eliminate data leakage.
- **Logistic Regression**: Yielded transparent **Odds Ratios ($\text{OR} = \exp(\beta)$)**:
  - `Fiber Optic Internet`: **$\text{OR} = 3.26\times$** (+226% churn odds vs DSL).
  - `Electronic Check`: **$\text{OR} = 1.48\times$** (+48% churn odds).
  - `Two-Year Contract`: **$\text{OR} = 0.27\times$** (-73% churn odds).
  - `tenure`: **$\text{OR} = 0.28\times$** per SD increase.
- **Random Forest Classifier**: Delivered **ROC-AUC = 0.8435** and **PR-AUC = 0.6587**.

---

## 📋 Four Strategic SaaS Retention Playbooks

### 📘 Playbook 1: The "First 90 Days" White-Glove Onboarding
- **Target**: New subscribers in Months 0–3 (where 62% churn occurs in Month 1).
- **Intervention**: Automated onboarding milestones on Days 1, 7, 21, and 45. Complimentary 90-day trial of **Tech Support & Online Security**. Outbound proactive outreach upon detecting >2 connection drops.
- **Impact**: Projected **25%–35% reduction in Month 1–3 drop-off**.

### 📘 Playbook 2: The "Commit & Save" Annual Contract Migration
- **Target**: 3,875 Month-to-Month subscribers (representing \$120.8K/mo in lost MRR).
- **Intervention**: Offer 1 month free or an 8% monthly discount for converting to a 12-month agreement.
- **Impact**: Converting 20% of Month-to-Month users (775 accounts) protects **\$288,000 in ARR**.

### 📘 Playbook 3: "Fiber Total Care" Support Architecture
- **Target**: 3,096 Fiber Optic subscribers (\$91.50/mo avg, 41.9% churn).
- **Intervention**: Eliminate bare standalone fiber plans. Automatically bundle `TechSupport` and `DeviceProtection` into fiber base pricing as "Fiber Total Care".
- **Impact**: Compresses fiber churn toward baseline DSL levels (~22%), preserving over **\$45,000/mo in MRR (\$540K ARR)**.

### 📘 Playbook 4: Automated Payment Migration (\$5 Bill Credit)
- **Target**: 2,365 customers using manual Electronic Checks (45.3% churn).
- **Intervention**: Provide a recurring \$5 monthly bill credit for setting up Credit Card or Bank Transfer Auto-Pay.
- **Impact**: Eliminates monthly active payment re-decision friction, reducing churn in this cohort by ~30 percentage points.

---

## 💰 Financial ROI & ARR Preservation Scenarios

| Churn Reduction Target | Accounts Retained Annually | Monthly MRR Preserved | Annual ARR Value Preserved | 3-Year Cumulative Value |
| :---: | :---: | :---: | :---: | :---: |
| **5% Reduction** | 93 accounts | **+\$6,956.54 / mo** | **+\$83,478.51 / yr** | \$250,435.53 |
| **10% Reduction** | 187 accounts | **+\$13,913.09 / mo** | **+\$166,957.02 / yr** | \$500,871.06 |
| **15% Reduction** | 280 accounts | **+\$20,869.63 / mo** | **+\$250,435.53 / yr** | \$751,306.59 |
| **20% Reduction** | 374 accounts | **+\$27,826.17 / mo** | **+\$333,914.04 / yr** | \$1,001,742.12 |

---

## 📜 License & Citation

- **License**: Apache 2.0
- **Dataset Source**: [IBM Sample Data / Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
