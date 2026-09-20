"""
Logistics Analytics Pipeline - Task 3: Advanced Analysis & Visualization
Uses Pandas, NumPy, and Matplotlib to analyze supply chain metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_summary_statistics(df):
    """
    Calculates central tendencies and variance for key metrics.
    """
    target_cols = [
        'Days for shipping (real)', 
        'Days for shipment (scheduled)', 
        'Sales per customer', 
        'Order Item Quantity'
    ]
    summary_df = df[target_cols].describe().T[['mean', '50%', 'std', 'min', 'max']]
    summary_df.rename(columns={'50%': 'median'}, inplace=True)
    return summary_df

def plot_shipping_delay_distribution(df):
    """
    Generates a histogram showing the distribution of shipping delays.
    """
    # Calculate delay variance
    delays = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    
    plt.figure(figsize=(8, 5))
    plt.hist(delays, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Delivery Delay Variance (Actual vs Scheduled)')
    plt.xlabel('Delay Duration (Days)')
    plt.ylabel('Shipment Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sales_vs_profit_correlation(df):
    """
    Generates a scatter plot comparing sales volume against order profit.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Sales per customer'], df['Order Profit Per Order'], alpha=0.5, color='darkblue')
    plt.title('Sales Volume vs. Order Profit Margin')
    plt.xlabel('Sales per Customer ($)')
    plt.ylabel('Order Profit ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_delivery_status_by_mode(df):
    """
    Generates a bar chart comparing shipping modes and delivery performance.
    """
    status_by_mode = pd.crosstab(df['Shipping Mode'], df['Delivery Status'])
    
    status_by_mode.plot(kind='bar', stacked=True, figsize=(9, 6), colormap='Set2')
    plt.title('Delivery Performance Breakdown by Shipping Mode')
    plt.xlabel('Shipping Mode')
    plt.ylabel('Number of Shipments')
    plt.xticks(rotation=0)
    plt.legend(title='Delivery Status')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Task 3 Analytics and Visualization Pipeline Module Ready.")
