import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --- THEME: EMERALD & VIOLET CYBER PALETTE ---
BG_COLOR = '#0f172a'      # Deep Slate Navy
PANEL_COLOR = '#1e293b'   # Card Background
TEXT_COLOR = '#f8fafc'    # Bright White
EMERALD = '#10b981'       # Positive / Retention
VIOLET = '#8b5cf6'        # Insights / Performance
CYAN = '#06b6d4'          # Trends
ROSE = '#f43f5e'          # Attrition / Risk

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI',
    'font.size': 8,
    'axes.edgecolor': CYAN,
    'axes.linewidth': 1.2,
    'figure.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': '#334155'
})

print("="*70)
print(" GENERATING VIBRANT HR ANALYTICS DASHBOARD (TASK 3)")
print("="*70)

# 1. Simulate HR & Employee Dataset
np.random.seed(202)
n_rows = 2200

emp_satisfaction = np.random.uniform(1.0, 5.0, size=n_rows)
monthly_income = np.random.uniform(25000, 150000, size=n_rows)
years_at_company = np.random.randint(1, 15, size=n_rows)
overtime = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
departments = np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], size=n_rows)

left_logits = 1.2 - (emp_satisfaction * 0.6) - (monthly_income / 50000) + (overtime * 1.4) - (years_at_company * 0.1)
left_prob = 1 / (1 + np.exp(-left_logits))
left_company = np.random.binomial(1, left_prob)

df_hr = pd.DataFrame({
    'Satisfaction': np.round(emp_satisfaction, 1),
    'Income': np.round(monthly_income, 2),
    'Years': years_at_company,
    'Overtime': overtime,
    'Department': departments,
    'Left': left_company
})

X = df_hr[['Satisfaction', 'Income', 'Years', 'Overtime']]
y = df_hr['Left']

# 2. Train Model for Insights
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_prob = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# 3. Create Grid Layout
fig = plt.figure(figsize=(19, 12), facecolor=BG_COLOR)
gs = fig.add_gridspec(4, 4, height_ratios=[0.6, 1.5, 1.5, 1.5], hspace=0.4, wspace=0.25)

# Header & KPI Cards
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')
ax_header.text(0.01, 0.7, "HR ANALYTICS & EMPLOYEE PERFORMANCE DASHBOARD", fontsize=20, fontweight='bold', color=CYAN)
ax_header.text(0.01, 0.2, "Monitor Workforce Retention | Department Attrition | Performance Drivers", fontsize=10, color=TEXT_COLOR)

ax_header.text(0.36, 0.5, f"Total Workforce\n{n_rows:,}", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=CYAN, color='white'))
ax_header.text(0.51, 0.5, f"Active Employees\n{n_rows - df_hr['Left'].sum()} ({(1-df_hr['Left'].mean())*100:.1f}%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=EMERALD, color='white'))
ax_header.text(0.66, 0.5, f"Attrition Count\n{df_hr['Left'].sum()} ({df_hr['Left'].mean()*100:.1f}%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=ROSE, color='white'))
ax_header.text(0.83, 0.5, f"Model ROC-AUC\n{roc_auc:.2f} (High)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=PANEL_COLOR, edgecolor=VIOLET, color='white'))

def style_ax(ax, title):
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(title, fontsize=10, fontweight='bold', color=CYAN, pad=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(CYAN)

# Row 1 Charts
ax_trend = fig.add_subplot(gs[1, 0:2])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
attrition_trend = [8.2, 7.9, 9.1, 8.5, 11.2, 10.4, 14.8, 12.1, 10.5, 9.4, 8.8, 8.0]
ax_trend.plot(months, attrition_trend, marker='o', color=VIOLET, linewidth=4, markersize=8, markerfacecolor=BG_COLOR, markeredgecolor=VIOLET)
style_ax(ax_trend, 'Monthly Attrition Rate Trend (%)')
ax_trend.grid(True, linestyle='-', alpha=0.5, color='#334155')

ax_donut = fig.add_subplot(gs[1, 2])
status_counts = [n_rows - df_hr['Left'].sum(), df_hr['Left'].sum()]
wedges, texts, autotexts = ax_donut.pie(status_counts, labels=['Retained', 'Left'], colors=[EMERALD, ROSE], autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops=dict(color=TEXT_COLOR, fontweight='bold'), wedgeprops=dict(width=0.45, edgecolor=PANEL_COLOR, linewidth=3))
for t in texts: t.set_color(TEXT_COLOR)
for t in autotexts: t.set_color('white')
style_ax(ax_donut, 'Workforce Retention Status')

ax_dept = fig.add_subplot(gs[1, 3])
dept_attrition = df_hr.groupby('Department')['Left'].mean() * 100
ax_dept.bar(dept_attrition.index, dept_attrition.values, color=[CYAN, EMERALD, VIOLET, ROSE, '#fbbf24'], width=0.6, edgecolor='white', linewidth=1)
style_ax(ax_dept, 'Attrition Rate by Department (%)')
ax_dept.tick_params(axis='x', rotation=15)

# Row 2 Charts
ax_tenure = fig.add_subplot(gs[2, 0])
tenure_attr = df_hr.groupby('Years')['Left'].mean() * 100
ax_tenure.plot(tenure_attr.index[:10], tenure_attr.values[:10], marker='s', color=EMERALD, linewidth=3, markersize=7)
style_ax(ax_tenure, 'Attrition vs Years at Company')
ax_tenure.set_xlabel('Years at Company', color=TEXT_COLOR)

ax_inc = fig.add_subplot(gs[2, 1])
df_hr['Income_Bracket'] = pd.qcut(df_hr['Income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
inc_attr = df_hr.groupby('Income_Bracket', observed=False)['Left'].mean() * 100
ax_inc.bar(inc_attr.index.astype(str), inc_attr.values, color=VIOLET, width=0.6, edgecolor='white', linewidth=1)
style_ax(ax_inc, 'Attrition by Income Bracket (%)')

ax_ot = fig.add_subplot(gs[2, 2])
ot_attr = df_hr.groupby('Overtime')['Left'].mean() * 100
ax_ot.bar(['No Overtime', 'Working Overtime'], ot_attr.values, color=[EMERALD, ROSE], width=0.5, edgecolor='white', linewidth=1)
style_ax(ax_ot, 'Impact of Overtime on Attrition (%)')

ax_reasons = fig.add_subplot(gs[2, 3])
reasons = ['Salary', 'Work-Life Balance', 'Growth', 'Management', 'Culture']
reason_pts = [42, 28, 16, 10, 4]
ax_reasons.barh(reasons, reason_pts, color=[ROSE, VIOLET, CYAN, EMERALD, '#777777'], edgecolor='white', linewidth=1)
style_ax(ax_reasons, 'Primary Reasons for Leaving')

# Row 3: Feature Importance & Employee Sample Table
ax_feat = fig.add_subplot(gs[3, 0:2])
features = ['Satisfaction', 'Income', 'Years', 'Overtime']
importances = rf_model.feature_importances_
indices = np.argsort(importances)
ax_feat.barh(np.array(features)[indices], importances[indices], color=CYAN, edgecolor='white', linewidth=1)
style_ax(ax_feat, 'Key Attrition Drivers (Model Insights)')
ax_feat.set_xlabel('Importance Score', color=TEXT_COLOR)

ax_table = fig.add_subplot(gs[3, 2:4])
ax_table.set_facecolor(PANEL_COLOR)
ax_table.axis('off')
sample_table_data = []
for i in range(5):
    sample_table_data.append([
        f'EMP_{501+i}', 
        df_hr['Department'].iloc[i],
        str(df_hr['Years'].iloc[i]), 
        f"₹{df_hr['Income'].iloc[i]:.0f}", 
        str(df_hr['Satisfaction'].iloc[i]), 
        'LEFT' if df_hr['Left'].iloc[i] == 1 else 'ACTIVE'
    ])

col_labels = ['Emp ID', 'Dept', 'Years', 'Income', 'Rating', 'Status']
table = ax_table.table(cellText=sample_table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    cell.set_facecolor(PANEL_COLOR)
    cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
    cell.set_edgecolor(CYAN)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(CYAN)
        cell.set_text_props(color='black', fontweight='bold')

ax_table.set_title('Employee Status Sample (Emerald/Violet Theme)', fontsize=10, fontweight='bold', color=CYAN)

plt.tight_layout()
output_filename = 'vibrant_powerbi_hr_dashboard.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print(f"\n[SUCCESS] Vibrant HR dashboard saved as '{output_filename}'.")
plt.show()