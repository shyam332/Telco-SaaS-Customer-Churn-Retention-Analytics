import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set global style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'

ARTIFACT_DIR = r"C:\Users\sahad\.gemini\antigravity\brain\38cedae6-c052-441a-867e-eb84732f5168"
SCRATCH_DIR = r"C:\Users\sahad\.gemini\antigravity\scratch\telco-customer-churn-analysis"
IMG_DIR = os.path.join(SCRATCH_DIR, "charts")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. Load Data
df = pd.read_csv(os.path.join(SCRATCH_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', '0.0'))
df['Churn_num'] = (df['Churn'] == 'Yes').astype(int)

# Color Palette
CHURN_COLOR = '#E63946'      # Energetic Red
RETAIN_COLOR = '#1D3557'     # Deep Navy
ACCENT_BLUE = '#457B9D'      # Slate Blue
ACCENT_TEAL = '#2A9D8F'      # Teal
MUTED_GRAY = '#6C757D'

# -------------------------------------------------------------------------
# CHART 1: Executive KPI & Revenue Churn Decomposition
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Donut 1: Customer Count
counts = df['Churn'].value_counts()
labels = [f"Retained\n({counts['No']:,} | {counts['No']/len(df):.1%})",
          f"Churned\n({counts['Yes']:,} | {counts['Yes']/len(df):.1%})"]
wedges1, texts1, autotexts1 = ax1.pie(
    counts, labels=labels, autopct='%1.1f%%', startangle=140,
    colors=[RETAIN_COLOR, CHURN_COLOR], explode=(0, 0.08),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    pctdistance=0.75
)
plt.setp(autotexts1, size=11, weight="bold", color="white")
ax1.set_title("Customer Churn Breakdown\n(Total Customers: 7,043)", pad=15)

# Donut 2: Monthly Recurring Revenue (MRR)
mrr_churn = df.groupby('Churn')['MonthlyCharges'].sum()
mrr_labels = [f"Retained MRR\n(${mrr_churn['No']:,.0f} | {mrr_churn['No']/df['MonthlyCharges'].sum():.1%})",
              f"Lost MRR\n(${mrr_churn['Yes']:,.0f} | {mrr_churn['Yes']/df['MonthlyCharges'].sum():.1%})"]
wedges2, texts2, autotexts2 = ax2.pie(
    mrr_churn, labels=mrr_labels, autopct='%1.1f%%', startangle=140,
    colors=[ACCENT_BLUE, CHURN_COLOR], explode=(0, 0.08),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
    pctdistance=0.75
)
plt.setp(autotexts2, size=11, weight="bold", color="white")
ax2.set_title(f"Monthly Revenue (MRR) Impact\n(Total MRR: ${df['MonthlyCharges'].sum():,.0f} | Lost ARR: ${mrr_churn['Yes']*12:,.0f})", pad=15)

plt.suptitle("Executive KPI Summary: Customer Volume vs. Revenue Lost", y=1.02)
plt.tight_layout()
chart1_path = os.path.join(IMG_DIR, "churn_kpi_revenue.png")
plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 2: Kaplan-Meier Survival / Retention Curves Across Tenure
# -------------------------------------------------------------------------
def kaplan_meier(durations, events):
    df_km = pd.DataFrame({'t': durations, 'e': events})
    unique_times = np.sort(df_km['t'].unique())
    survival = []
    s = 1.0
    current_at_risk = len(df_km)
    
    for t in unique_times:
        d = ((df_km['t'] == t) & (df_km['e'] == 1)).sum()
        c = ((df_km['t'] == t) & (df_km['e'] == 0)).sum()
        if current_at_risk > 0:
            s *= (1.0 - d / current_at_risk)
        survival.append(s)
        current_at_risk -= (d + c)
    return unique_times, np.array(survival)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Overall KM Curve
t_all, s_all = kaplan_meier(df['tenure'], df['Churn_num'])
ax1.plot(t_all, s_all * 100, color=RETAIN_COLOR, lw=3, label='Overall Customer Base')
ax1.fill_between(t_all, s_all * 100, 100, color=CHURN_COLOR, alpha=0.15, label='Cumulative Churn Area')
ax1.axvline(12, color=CHURN_COLOR, linestyle='--', lw=1.5, label='Month 12 Milestone (84.3% retention)')
ax1.axvline(24, color='orange', linestyle='--', lw=1.5, label='Month 24 Milestone (78.9% retention)')
ax1.set_title("Customer Retention Survival Curve (0-72 Months)")
ax1.set_xlabel("Tenure (Months on Platform)")
ax1.set_ylabel("Survival Probability (% Retained)")
ax1.set_ylim(40, 102)
ax1.legend(loc='lower left', frameon=True)

# Annotate month 1 drop
ax1.annotate('Month 1 Cliff:\n380 churns (62% of M1 exits!)', 
             xy=(1, 94.6), xytext=(8, 88),
             arrowprops=dict(facecolor=CHURN_COLOR, shrink=0.08, width=1.5, headwidth=8),
             fontweight='bold', color=CHURN_COLOR)

# By Contract Curve
contract_colors = {'Month-to-month': CHURN_COLOR, 'One year': ACCENT_BLUE, 'Two year': ACCENT_TEAL}
for c_name, color in contract_colors.items():
    sub = df[df['Contract'] == c_name]
    t_c, s_c = kaplan_meier(sub['tenure'], sub['Churn_num'])
    ax2.plot(t_c, s_c * 100, color=color, lw=3, label=f"{c_name} (S(24m)={s_c[t_c<=24][-1]*100:.1f}%)")

ax2.set_title("Retention Curve by Contract Cohort")
ax2.set_xlabel("Tenure (Months on Platform)")
ax2.set_ylabel("Survival Probability (% Retained)")
ax2.set_ylim(30, 102)
ax2.legend(loc='lower left', frameon=True)

plt.suptitle("Tenure Dynamics & Cohort Retention Survival Analysis", y=1.02)
plt.tight_layout()
chart2_path = os.path.join(IMG_DIR, "survival_curves.png")
plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 3: Contract & Payment Friction Analysis
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Contract Churn Rate & Customer Volume
contract_order = ['Month-to-month', 'One year', 'Two year']
c_stats = df.groupby('Contract').agg(
    churn_rate=('Churn_num', 'mean'),
    volume=('customerID', 'count')
).reindex(contract_order)

bars1 = ax1.bar(c_stats.index, c_stats['churn_rate'] * 100, color=[CHURN_COLOR, ACCENT_BLUE, ACCENT_TEAL], width=0.55, edgecolor='black', lw=0.5)
ax1.set_title("Churn Rate by Contract Commitment Type")
ax1.set_ylabel("Churn Rate (%)")
ax1.set_ylim(0, 52)
for bar, vol in zip(bars1, c_stats['volume']):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%\n({vol:,} users)", ha='center', va='bottom', fontweight='bold')

# Payment Method Churn Rate & Volume
pm_order = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
pm_stats = df.groupby('PaymentMethod').agg(
    churn_rate=('Churn_num', 'mean'),
    volume=('customerID', 'count')
).reindex(pm_order)

pm_colors = [CHURN_COLOR, '#F4A261', ACCENT_BLUE, ACCENT_TEAL]
bars2 = ax2.barh(pm_stats.index, pm_stats['churn_rate'] * 100, color=pm_colors, height=0.55, edgecolor='black', lw=0.5)
ax2.set_title("Churn Rate by Payment Method (Friction Driver)")
ax2.set_xlabel("Churn Rate (%)")
ax2.set_xlim(0, 55)
for bar, vol in zip(bars2, pm_stats['volume']):
    xval = bar.get_width()
    ax2.text(xval + 1.0, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f}% ({vol:,} users)", ha='left', va='center', fontweight='bold')

plt.suptitle("Structural Drivers: Contract Type & Payment Method Friction", y=1.02)
plt.tight_layout()
chart3_path = os.path.join(IMG_DIR, "contract_payment_friction.png")
plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 4: Service Ecosystem & Add-on Shield Analysis
# -------------------------------------------------------------------------
services = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']
services_data = []

for s in services:
    sub_df = df[df[s] != 'No internet service']
    rates = sub_df.groupby(s)['Churn_num'].mean() * 100
    services_data.append({
        'Service': s,
        'Without Add-on': rates.get('No', 0),
        'With Add-on': rates.get('Yes', 0),
        'Churn Reduction': rates.get('No', 0) - rates.get('Yes', 0)
    })

serv_df = pd.DataFrame(services_data).sort_values('Churn Reduction', ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(serv_df))
bar_width = 0.35

rects1 = ax.barh(y_pos - bar_width/2, serv_df['Without Add-on'], bar_width, label='Without Service Add-on', color=CHURN_COLOR, edgecolor='black', lw=0.5)
rects2 = ax.barh(y_pos + bar_width/2, serv_df['With Add-on'], bar_width, label='With Service Add-on', color=ACCENT_TEAL, edgecolor='black', lw=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(serv_df['Service'], fontweight='bold')
ax.set_xlabel("Churn Rate (%)")
ax.set_title("The 'Retention Shield': Impact of Technical & Support Add-ons on Churn")
ax.set_xlim(0, 50)
ax.legend(loc='lower right', frameon=True)

for i, (w_out, w_in, red) in enumerate(zip(serv_df['Without Add-on'], serv_df['With Add-on'], serv_df['Churn Reduction'])):
    ax.text(w_out + 0.8, i - bar_width/2, f"{w_out:.1f}%", va='center', fontsize=9, fontweight='bold', color=CHURN_COLOR)
    ax.text(w_in + 0.8, i + bar_width/2, f"{w_in:.1f}% (-{red:.1f}%)", va='center', fontsize=9, fontweight='bold', color=ACCENT_TEAL)

plt.tight_layout()
chart4_path = os.path.join(IMG_DIR, "service_ecosystem_shield.png")
plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 5: Pricing Elasticity & Bill Shock Distribution
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Distribution of Monthly Charges by Churn
sns.kdeplot(data=df[df['Churn']=='No']['MonthlyCharges'], ax=ax1, fill=True, color=RETAIN_COLOR, label='Retained (Median: $64.4)', lw=2.5)
sns.kdeplot(data=df[df['Churn']=='Yes']['MonthlyCharges'], ax=ax1, fill=True, color=CHURN_COLOR, label='Churned (Median: $79.7)', lw=2.5)
ax1.set_title("Monthly Charges Density: Retained vs. Churned")
ax1.set_xlabel("Monthly Charges ($)")
ax1.set_ylabel("Customer Density")
ax1.legend(loc='upper right', frameon=True)

# Churn rate by Internet Service & Monthly Charges Tier
df['Charge_Tier'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Q1 Low ($18-35)', 'Q2 Mid-Low ($35-70)', 'Q3 Mid-High ($70-90)', 'Q4 High ($90-119)'])
tier_churn = df.groupby(['InternetService', 'Charge_Tier'], observed=False)['Churn_num'].mean().unstack() * 100

tier_churn.plot(kind='bar', ax=ax2, colormap='Blues', edgecolor='black', lw=0.5, width=0.7)
ax2.set_title("Churn Rate by Internet Service & Pricing Tier")
ax2.set_xlabel("Internet Service Architecture")
ax2.set_ylabel("Churn Rate (%)")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontweight='bold')
ax2.legend(title='Monthly Charge Tier', frameon=True)

plt.suptitle("Price Sensitivity & Fiber Optic Pricing Friction", y=1.02)
plt.tight_layout()
chart5_path = os.path.join(IMG_DIR, "pricing_elasticity_shock.png")
plt.savefig(chart5_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 6: Statistical Drivers & ML Feature Importance
# -------------------------------------------------------------------------
X = df.drop(columns=['customerID', 'Churn', 'Churn_num', 'Charge_Tier'])
y = df['Churn_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_proc, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_proc, y_train)

# Plot Feature Importances
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Odds Ratios
odds_df = pd.DataFrame({'Feature': all_feature_names, 'Odds_Ratio': np.exp(lr.coef_[0])})
# Clean labels for readability
odds_df['Clean_Feature'] = odds_df['Feature'].str.replace('InternetService_', 'Internet: ').str.replace('PaymentMethod_', 'Payment: ').str.replace('Contract_', 'Contract: ')
top_bottom_odds = pd.concat([odds_df.sort_values('Odds_Ratio', ascending=False).head(6),
                             odds_df.sort_values('Odds_Ratio', ascending=True).head(6)]).drop_duplicates()
top_bottom_odds = top_bottom_odds.sort_values('Odds_Ratio', ascending=True)

colors_or = [ACCENT_TEAL if o < 1.0 else CHURN_COLOR for o in top_bottom_odds['Odds_Ratio']]
ax1.barh(top_bottom_odds['Clean_Feature'], top_bottom_odds['Odds_Ratio'], color=colors_or, edgecolor='black', lw=0.5, height=0.6)
ax1.axvline(1.0, color='gray', linestyle='--', lw=1.5)
ax1.set_title("Logistic Regression: Churn Odds Ratios\n(Values > 1.0 increase churn risk; < 1.0 protect retention)")
ax1.set_xlabel("Odds Ratio (Multiplicative Effect on Churn Odds)")

for bar in ax1.patches:
    val = bar.get_width()
    offset = 0.05 if val >= 1.0 else -0.2
    ax1.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:.2f}x", va='center', fontweight='bold', fontsize=9)

# Random Forest Gini Importance
rf_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': rf.feature_importances_})
rf_df['Clean_Feature'] = rf_df['Feature'].str.replace('InternetService_', 'Internet: ').str.replace('PaymentMethod_', 'Payment: ').str.replace('Contract_', 'Contract: ')
top_rf = rf_df.sort_values('Importance', ascending=True).tail(10)

ax2.barh(top_rf['Clean_Feature'], top_rf['Importance'] * 100, color=ACCENT_BLUE, edgecolor='black', lw=0.5, height=0.6)
ax2.set_title("Random Forest: Top 10 Feature Importances\n(Gini Impurity Contribution %)")
ax2.set_xlabel("Relative Importance (%)")

for bar in ax2.patches:
    val = bar.get_width()
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center', fontweight='bold', fontsize=9)

plt.suptitle("Statistical & Machine Learning Churn Driver Identification", y=1.02)
plt.tight_layout()
chart6_path = os.path.join(IMG_DIR, "ml_drivers_importance.png")
plt.savefig(chart6_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# CHART 7: Cohort Retention Matrix Heatmap
# -------------------------------------------------------------------------
bins = [-1, 1, 6, 12, 24, 48, 72]
labels = ['0-1 mo', '2-6 mo', '7-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
df['Tenure_Cohort'] = pd.cut(df['tenure'], bins=bins, labels=labels)

pivot_cohort = df.pivot_table(
    index='Contract',
    columns='Tenure_Cohort',
    values='Churn_num',
    aggfunc='mean',
    observed=False
) * 100

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot_cohort, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Churn Rate (%)'}, 
            linewidths=1.5, linecolor='white', ax=ax, annot_kws={"weight": "bold", "size": 12})
ax.set_title("Cohort Churn Matrix: Contract Type vs. Tenure Duration (%)", pad=15)
ax.set_xlabel("Customer Tenure Cohort")
ax.set_ylabel("Contract Model")
plt.tight_layout()
chart7_path = os.path.join(IMG_DIR, "cohort_retention_heatmap.png")
plt.savefig(chart7_path, dpi=300, bbox_inches='tight')
plt.close()

# Copy all charts to the artifact directory so artifacts can link them
for chart_name in os.listdir(IMG_DIR):
    src = os.path.join(IMG_DIR, chart_name)
    dst = os.path.join(ARTIFACT_DIR, chart_name)
    shutil.copy2(src, dst)

print("All 7 publication-quality charts successfully created and synced to artifact directory!")
