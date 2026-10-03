import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Rich Blue Background & High-Vibrant Contrast Configuration for Task 1
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI',
    'font.size': 8,
    'axes.edgecolor': '#38bdf8',
    'axes.linewidth': 1.0,
    'figure.facecolor': '#0b192c'
})

print("="*70)
print(" GENERATING VIBRANT POWER BI STYLE SALES DASHBOARD (TASK 1)")
print("="*70)

# Simulate Business Sales Dataset
np.random.seed(101)
n_rows = 1000

categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Sports']
regions = ['North', 'South', 'East', 'West', 'Central']

category_data = np.random.choice(categories, size=n_rows)
region_data = np.random.choice(regions, size=n_rows)
units_sold = np.random.randint(1, 15, size=n_rows)
unit_price = np.random.uniform(100.0, 2500.0, size=n_rows)
sales_amount = np.round(units_sold * unit_price, 2)

df_sales = pd.DataFrame({
    'Category': category_data,
    'Region': region_data,
    'Units_Sold': units_sold,
    'Sales_Amount': sales_amount
})

cat_sales = df_sales.groupby('Category')['Sales_Amount'].sum().reset_index()
reg_sales = df_sales.groupby('Region')['Sales_Amount'].sum().reset_index()

# --- Create Vibrant Grid Layout for Sales ---
fig = plt.figure(figsize=(18, 11), facecolor='#0b192c')
gs = fig.add_gridspec(4, 4, height_ratios=[0.5, 1.5, 1.5, 1.5], hspace=0.45, wspace=0.28)

# 1. Header Banner & Glowing KPI Cards
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')
ax_header.text(0.01, 0.7, "BUSINESS SALES PERFORMANCE DASHBOARD", fontsize=18, fontweight='bold', color='#ffffff')
ax_header.text(0.01, 0.2, "Monitor Revenue Growth  |  Analyze Regional Performance  |  Track Top Product Categories", fontsize=9, color='#38bdf8')

ax_header.text(0.38, 0.5, "Total Revenue\n₹4.82M (+12.4%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e3a8a', edgecolor='#38bdf8', color='white'))
ax_header.text(0.53, 0.5, "Units Sold\n24,850 (+8.1%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#065f46', edgecolor='#34d399', color='white'))
ax_header.text(0.68, 0.5, "Avg Order Value\n₹1,940 (+3.2%)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#831843', edgecolor='#f43f5e', color='white'))
ax_header.text(0.85, 0.5, "Profit Margin\n24.6% (High)", fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', facecolor='#581c87', edgecolor='#c084fc', color='white'))

# Helper function for glowing blue panels & bright text
def style_ax(ax, title):
    ax.set_facecolor('#1e293b')
    ax.set_title(title, fontsize=9, fontweight='bold', color='#38bdf8', pad=8)
    ax.tick_params(colors='#ffffff', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#38bdf8')

# 2. Row 1 Charts: Monthly Trend, Category Share Donut, Distribution Channel
ax_trend = fig.add_subplot(gs[1, 0:2])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_rev = [320, 340, 310, 390, 420, 460, 440, 510, 490, 530, 580, 620]
ax_trend.plot(months, monthly_rev, marker='o', color='#facc15', linewidth=3, markersize=6)
style_ax(ax_trend, 'Monthly Sales Revenue Trend (₹ in Thousands)')
ax_trend.grid(True, linestyle=':', alpha=0.4, color='#38bdf8')

ax_donut = fig.add_subplot(gs[1, 2])
cat_shares = [35, 25, 20, 12, 8]
wedges, texts, autotexts = ax_donut.pie(cat_shares, labels=categories, colors=['#38bdf8', '#f43f5e', '#fbbf24', '#c084fc', '#34d399'], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.45, edgecolor='#1e293b', linewidth=2))
for t in texts + autotexts:
    t.set_color('white')
    t.set_fontweight('bold')
style_ax(ax_donut, 'Sales Share by Category')

ax_sub = fig.add_subplot(gs[1, 3])
channels = ['Online Store', 'Direct B2B', 'Retail Partners', 'Mobile App', 'Wholesale']
channel_sales = [45, 28, 15, 8, 4]
ax_sub.bar(channels, channel_sales, color=['#f43f5e', '#fbbf24', '#38bdf8', '#c084fc', '#34d399'], width=0.55, edgecolor='white', linewidth=0.8)
style_ax(ax_sub, 'Sales by Distribution Channel')
ax_sub.tick_params(axis='x', rotation=20)

# 3. Row 2 Charts: Category Performance, Regional Sales, Discount Impact, Purchase Drivers
ax_cat = fig.add_subplot(gs[2, 0])
sns.barplot(ax=ax_cat, x='Category', y='Sales_Amount', data=cat_sales, palette=['#38bdf8', '#f43f5e', '#fbbf24', '#c084fc', '#34d399'], hue='Category', legend=False, edgecolor='white')
style_ax(ax_cat, 'Total Sales by Category')
ax_cat.set_ylabel('Sales Amount (₹)', color='#ffffff')
ax_cat.set_xlabel('')
ax_cat.tick_params(axis='x', rotation=20)

ax_reg = fig.add_subplot(gs[2, 1])
sns.barplot(ax=ax_reg, x='Region', y='Sales_Amount', data=reg_sales, palette='spring', hue='Region', legend=False, edgecolor='white')
style_ax(ax_reg, 'Regional Sales Performance')
ax_reg.set_ylabel('Sales Amount (₹)', color='#ffffff')
ax_reg.set_xlabel('')

ax_disc = fig.add_subplot(gs[2, 2])
disc_bins = ['0%', '5%', '10%', '15%', '20%+']
disc_sales = [120, 250, 410, 310, 150]
ax_disc.bar(disc_bins, disc_sales, color='#c084fc', width=0.55, edgecolor='white', linewidth=0.8)
style_ax(ax_disc, 'Sales Volume by Discount Tier')
ax_disc.set_xlabel('Discount Level', color='#ffffff')

ax_reasons = fig.add_subplot(gs[2, 3])
drivers = ['Quality', 'Price', 'Delivery', 'Support', 'Brand']
driver_pts = [42, 28, 16, 8, 6]
ax_reasons.barh(drivers, driver_pts, color='#fb7185', edgecolor='white', linewidth=0.8)
style_ax(ax_reasons, 'Key Drivers of Purchase')

# 4. Row 3: Top Products & Transaction Sample Table
ax_feat = fig.add_subplot(gs[3, 0:2])
top_products = ['Smart TV 4K', 'Wireless Earbuds', 'Running Shoes', 'Smart Watch', 'Blender Pro']
prod_rev = [0.38, 0.28, 0.18, 0.11, 0.05]
ax_feat.barh(top_products, prod_rev, color='#38bdf8', edgecolor='white', linewidth=0.8)
style_ax(ax_feat, 'Top Revenue Generating Products')
ax_feat.set_xlabel('Revenue Share Score', color='#ffffff')

ax_table = fig.add_subplot(gs[3, 2:4])
ax_table.set_facecolor('#1e293b')
ax_table.axis('off')
table_data = [
    ['TXN_901', 'Electronics', 'North', '2', '₹45,000', 'Completed'],
    ['TXN_902', 'Apparel', 'South', '4', '₹6,400', 'Completed'],
    ['TXN_903', 'Home', 'West', '1', '₹12,500', 'Pending'],
    ['TXN_904', 'Sports', 'East', '3', '₹9,800', 'Completed']
]
col_labels = ['Txn ID', 'Category', 'Region', 'Units', 'Amount', 'Status']
table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.3)
for key, cell in table.get_celld().items():
    cell.set_facecolor('#0b192c')
    cell.set_text_props(color='#ffffff', fontweight='bold')
    cell.set_edgecolor('#38bdf8')
ax_table.set_title('Recent Sales Transactions', fontsize=9, fontweight='bold', color='#38bdf8')

plt.tight_layout()
output_filename = 'vibrant_powerbi_sales_dashboard.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='#0b192c')
print(f"\n[SUCCESS] Vibrant Sales dashboard saved as '{output_filename}'.")
plt.show()