#!/usr/bin/env python
# coding: utf-8

# # 1. Data Loading & Initial Exploration

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
df = pd.read_csv('/Users/Dell/OneDrive/Desktop/Ecommerce_Strategic_Assignment_Dataset.csv')

# Display first 5 rows
print("First 5 rows:")
display(df.head())

# Check basic info
print("\nDataset info:")
df.info()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Basic statistics
print("\nDescriptive statistics:")
display(df.describe())


# # 2. Data Cleaning

# In[6]:


# Fill missing numerical values with median
df['Marketing_Spend'] = df['Marketing_Spend'].fillna(df['Marketing_Spend'].median())
df['Revenue'] = df['Revenue'].fillna(df['Revenue'].median())
df['Purchases'] = df['Purchases'].fillna(df['Purchases'].median())

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])


# # 3. Key Performance Metrics
# 

# ## A. Conversion Rate by Traffic Source

# In[7]:


conversion_by_source = df.groupby('Traffic_Source')['Conversion_Rate'].mean().sort_values(ascending=False)
print("Avg. Conversion Rate by Traffic Source:\n", conversion_by_source)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=conversion_by_source.index, y=conversion_by_source.values)
plt.title("Conversion Rate by Traffic Source")
plt.ylabel("Conversion Rate (%)")
plt.xticks(rotation=45)
plt.show()


# ## B. Revenue vs. Marketing Spend

# In[16]:


# Calculate ROI by Campaign
roi_by_campaign = df.groupby('Campaign').apply(
    lambda x: (x['Revenue'].sum() - x['Marketing_Spend'].sum()) / x['Marketing_Spend'].sum()
).sort_values(ascending=False)

# Calculate ROI by Traffic Source
roi_by_traffic = df.groupby('Traffic_Source').apply(
    lambda x: (x['Revenue'].sum() - x['Marketing_Spend'].sum()) / x['Marketing_Spend'].sum()
).sort_values(ascending=False)

print("ROI by Campaign:\n", roi_by_campaign)
print("\nROI by Traffic Source:\n", roi_by_traffic)

# Create figure with two subplots
plt.figure(figsize=(16, 6))

# ROI by Campaign plot
plt.subplot(1, 2, 1)
sns.barplot(x=roi_by_campaign.index, y=roi_by_campaign.values, palette="Blues_d")
plt.title("Return on Investment (ROI) by Campaign", pad=20)
plt.ylabel("ROI (Revenue/Spend)")
plt.xticks(rotation=45)
plt.axhline(y=1, color='red', linestyle='--', linewidth=1)  # Break-even line

# ROI by Traffic Source plot
plt.subplot(1, 2, 2)
sns.barplot(x=roi_by_traffic.index, y=roi_by_traffic.values, palette="Greens_d")
plt.title("Return on Investment (ROI) by Traffic Source", pad=20)
plt.ylabel("ROI (Revenue/Spend)")
plt.xticks(rotation=45)
plt.axhline(y=1, color='red', linestyle='--', linewidth=1)  # Break-even line

plt.tight_layout()
plt.show()


# ## C. Customer Retention Analysis
# 

# In[10]:


repeat_purchase_rates = df.groupby('Customer_Type')['Repeat_Purchase_Rate'].mean()
print("Avg. Repeat Purchase Rate:\n", repeat_purchase_rates)

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(x=repeat_purchase_rates.index, y=repeat_purchase_rates.values)
plt.title("Repeat Purchase Rate by Customer Type")
plt.ylabel("Repeat Purchase Rate (%)")
plt.show()


# ## D. Cart Abandonment Analysis
# 

# In[11]:


abandonment_by_source = df.groupby('Traffic_Source')['Cart_Abandonment_Rate'].mean().sort_values(ascending=False)
print("Avg. Cart Abandonment by Source:\n", abandonment_by_source)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=abandonment_by_source.index, y=abandonment_by_source.values)
plt.title("Cart Abandonment Rate by Traffic Source")
plt.ylabel("Abandonment Rate (%)")
plt.xticks(rotation=45)
plt.show()


# # 4. Advanced Insights (Correlation & Trends)

# ## A. Correlation Matrix
# 

# In[17]:


corr_matrix = df[['Visitors', 'Marketing_Spend', 'Add_to_Cart', 'Purchases', 'Revenue']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# ## B. Monthly Revenue Trend
# 

# In[19]:


df['Month'] = df['Date'].dt.month
monthly_revenue = df.groupby('Month')['Revenue'].sum()

plt.figure(figsize=(8, 4))
sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values, marker='o')
plt.grid(True, linestyle='--', alpha=0.6)

for x, y in zip(monthly_revenue.index, monthly_revenue.values):
    plt.text(x, y, f"${y:,.0f}", ha='center', va='bottom', fontsize=9)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue ($)")
plt.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
plt.tight_layout()
plt.show()


# In[20]:


# Group by traffic source
traffic_stats = df.groupby('Traffic_Source').agg({
    'Visitors': 'sum',
    'Marketing_Spend': 'sum',
    'Revenue': 'sum',
    'Conversion_Rate': 'mean',
    'Cart_Abandonment_Rate': 'mean'
}).sort_values('Revenue', ascending=False)

print(traffic_stats)


# In[ ]:




