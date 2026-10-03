import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# --- NEW COLOR PALETTE: SUNSET AMBER & TEAL ---
# Dark Charcoal Background for depth
BG_COLOR = '#121212'  
PANEL_COLOR = '#1e1e1e' 
TEXT_COLOR = '#e0e0e0'
# Accent Colors: Amber (Trend), Teal (Category), Violet (Metric), Red (Churn)
AMBER = '#ffb703'
TEAL = '#2a9d8f'
VIOLET = '#9b5de5'
RED_CHURN = '#ef476f'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI',
    'font.size': 8,
    'axes.edgecolor': AMBER,
    'axes.linewidth': 1.2,
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': '#333333'
})

print("="*70)
print(" GENERATING VIBRANT CHURN DASHBOARD (TASK 2 - AMBER/TEAL THEME)")
print("="*70)

# 1. Simulate & Prepare Churn Dataset
np.random.seed(102) # Changed seed for variation
n_rows = 2500

tenure = np.random.randint(1, 72, size=n_rows)
monthly_charges = np.random.uniform(30.0, 160.0, size=n_rows)
support_tickets = np.random.poisson(lam=1.8, size=n_rows)
contract_type = np.random.choice([0, 1, 2], size=n_rows, p=[0.6, 0.25, 0.15]) # 0: Month-to-Month, 1: One Year, 2: Two Year
regions = np.random.choice(['Metro', 'Urban', 'Rural'], size=n_rows, p=[0.4, 0.35, 0.25])

churn_logits = -2.5 + (contract_type == 0) * 2.1 + support_tickets * 0.38 - tenure * 0.05
churn_prob_sigmoid = 1 / (1 + np.exp(-churn_logits))
churned = np.random.binomial(1, churn_prob_sigmoid)

df = pd.DataFrame({
    'Tenure': tenure,
    'Monthly_Charges': np.round(monthly_charges, 2),
    'Support_Tickets': support_tickets,
    'Contract_Type': contract_type,
    'Region': regions,
    'Churn': churned
})

X = df[['Tenure', 'Monthly_Charges', 'Support_Tickets', 'Contract_Type']]
y = df['Churn']

# 2. Train Random Forest Model for Insights
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"[MODEL] Model trained. ROC-AUC Score: {roc_auc:.4f}")

# 3. --- Create Vibrant Grid Layout with New Palette ---
fig = plt.figure(figsize=(19, 12), facecolor=BG_COLOR)
gs = fig.add_gridspec(4, 4, height_ratios=[0.6, 1.5, 1.5, 1.5], hspace=0.4, wspace=0.25)

# Header Banner & KPI Cards (Styled with new accents)
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')
ax_header.text(0.01, 0.7, "ADVANCED CHURN ANALYTICS DASHBOARD", fontsize=20, fontweight='bold', color=AMBER)
ax_header.text(0.01, 0.2, "Predictive Insights into Customer Retention | Regions: Metro, Urban, Rural", fontsize=10, color=TEXT_COLOR)

ax_header.text(0.36, 0.5, f"Total\nCustomers\n{n_rows:,}", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=TEAL, color='white'))
ax_header.text(0.51, 0.5, f"Churned\nCount\n{df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=RED_CHURN, color='white'))
ax_header.text(0.66, 0.5, f"Active\nCustomers\n{n_rows - df['Churn'].sum()} ({(1-df['Churn'].mean())*100:.1f}%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=TEAL, color='white'))
ax_header.text(0.83, 0.5, f"Model\nROC-AUC\n{roc_auc:.2f}", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=VIOLET, color='white'))

def style_ax(ax, title):
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(title, fontsize=10, fontweight='bold', color=AMBER, pad=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(TEAL)

# Row 1: Churn Trend (Amber), Status Donut (Teal/Red), Churn by Contract
ax_trend = fig.add_subplot(gs[1, 0:2])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_churn_rate = [12.1, 11.8, 13.5, 12.1, 15.2, 14.0, 23.5, 18.4, 16.8, 15.2, 14.0, 13.1]
ax_trend.plot(months, monthly_churn_rate, marker='o', color=AMBER, linewidth=4, markersize=8, markerfacecolor=BG_COLOR, markeredgecolor=AMBER)
style_ax(ax_trend, 'Monthly Churn Rate Trend (%)')
ax_trend.grid(True, linestyle='-', alpha=0.5, color='#333333')

ax_donut = fig.add_subplot(gs[1, 2])
status_counts = [n_rows - df['Churn'].sum(), df['Churn'].sum()]
wedges, texts, autotexts = ax_donut.pie(status_counts, labels=['Active', 'Churned'], colors=[TEAL, RED_CHURN], autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops=dict(color=TEXT_COLOR, fontweight='bold'), wedgeprops=dict(width=0.45, edgecolor=PANEL_COLOR, linewidth=3))
for t in texts: t.set_color(TEXT_COLOR)
for t in autotexts: t.set_color('white')
style_ax(ax_donut, 'Customer Status Distribution')
ax_donut.text(0, 0, 'TOTAL', fontsize=8, ha='center', va='center', color=TEXT_COLOR)

ax_sub = fig.add_subplot(gs[1, 3])
contract_churn = df.groupby('Contract_Type')['Churn'].mean() * 100
ax_sub.bar(['Month-to-Month', 'One Year', 'Two Year'], contract_churn, color=[RED_CHURN, AMBER, TEAL], width=0.6, edgecolor='white', linewidth=1)
style_ax(ax_sub, 'Churn Rate by Contract Type (%)')

# Row 2: Churn by Tenure (Violet), Regional Churn (Teal), Churn vs Support
ax_tenure = fig.add_subplot(gs[2, 0])
df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs'])
tenure_churn = df.groupby('Tenure_Group', observed=False)['Churn'].mean() * 100
ax_tenure.bar(tenure_churn.index.astype(str), tenure_churn.values, color=VIOLET, width=0.6, edgecolor='white', linewidth=1)
style_ax(ax_tenure, 'Churn Rate by Tenure Group (%)')

ax_reg = fig.add_subplot(gs[2, 1])
reg_churn = df.groupby('Region')['Churn'].mean() * 100
sns.barplot(ax=ax_reg, x=reg_churn.index, y=reg_churn.values, palette='viridis', hue=reg_churn.index, legend=False, edgecolor='white', linewidth=1)
style_ax(ax_reg, 'Churn Rate by Region (%)')
ax_reg.set_ylabel('Churn %', color=TEXT_COLOR)
ax_reg.set_xlabel('')

ax_sup = fig.add_subplot(gs[2, 2])
ticket_churn = df.groupby('Support_Tickets')['Churn'].mean() * 100
ax_sup.plot(ticket_churn.index[:8], ticket_churn.values[:8], marker='D', color=TEAL, linewidth=3, markersize=7, markerfacecolor=PANEL_COLOR, markeredgecolor=TEAL)
style_ax(ax_sup, 'Churn Rate vs Support Tickets')
ax_sup.set_xlabel('Support Tickets Count', color=TEXT_COLOR)

ax_reasons = fig.add_subplot(gs[2, 3])
reasons = ['Poor Service', 'High Prices', 'Better Competitor', 'Network Issues', 'Feature Gap']
reason_pts = [45, 30, 15, 7, 3]
ax_reasons.barh(reasons, reason_pts, color=[RED_CHURN, AMBER, TEAL, VIOLET, '#777777'], edgecolor='white', linewidth=1)
style_ax(ax_reasons, 'Top Reasons for Customer Churn (Survey)')

# Row 3: Feature Importance (Teal) & Prediction Sample Table
ax_feat = fig.add_subplot(gs[3, 0:2])
features = ['Tenure', 'Monthly Charges', 'Support Tickets', 'Contract Type']
importances = rf_model.feature_importances_
indices = np.argsort(importances)
ax_feat.barh(np.array(features)[indices], importances[indices], color=TEAL, edgecolor='white', linewidth=1)
style_ax(ax_feat, 'Key Feature Importance (Model Insights)')
ax_feat.set_xlabel('Importance Score', color=TEXT_COLOR)

ax_table = fig.add_subplot(gs[3, 2:4])
ax_table.set_facecolor(PANEL_COLOR)
ax_table.axis('off')
sample_table_data = []
for i in range(5):
    sample_table_data.append([
        f'CUST_{1001+i}', 
        str(df['Tenure'].iloc[i]), 
        f"₹{df['Monthly_Charges'].iloc[i]:.0f}", 
        str(df['Support_Tickets'].iloc[i]), 
        f"{y_prob[i]:.2f}", 
        'CHURNED' if y_pred[i] == 1 else 'ACTIVE'
    ])

col_labels = ['Cust ID', 'Tenure', 'Charges', 'Tickets', 'Prob', 'Status']
table = ax_table.table(cellText=sample_table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    cell.set_facecolor(PANEL_COLOR)
    cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
    cell.set_edgecolor(TEAL)
# Style header row of table
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(AMBER)
        cell.set_text_props(color='black', fontweight='bold')

ax_table.set_title('Churn Prediction Sample (Amber/Teal Theme)', fontsize=10, fontweight='bold', color=AMBER)

plt.tight_layout()
output_filename = 'vibrant_powerbi_churn_dashboard_amber_teal.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print(f"\n[SUCCESS] Vibrant Churn dashboard (Amber/Teal) saved as '{output_filename}'.")
plt.show()