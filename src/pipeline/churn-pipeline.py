# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:20:10 2025

@author: ngozi
"""

# Commit 1: Initialize customer churn pipeline with data loading
# Task: Load the customer analytics CSV file and display basic dataset information
# Goal: Create a simple script that reads CSV, shows shape, columns, and sample data

import pandas as pd
#import numpy as np

def load_customer_data(file_path='data/synthetic/customer_analytics_master.csv'):
    """
    Load customer analytics data from CSV file.
    Returns pandas DataFrame with basic info printed.
    """
    print("🔄 Loading customer analytics data...")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("✅ Data loaded successfully!")
    print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🏷️  Columns: {list(df.columns)}")
    
    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Show data types
    print("\n📋 Data types:")
    print(df.dtypes)
    
    # Show first few rows
    print("\n👀 First 3 rows:")
    print(df.head(3))
    
    # Basic stats
    print("\n📈 Basic info:")
    print(f"   • Unique customers: {df['customer_id'].nunique():,}")
    print(f"   • Unique orders: {df['order_id'].nunique():,}")
    print(f"   • Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    
    return df

if __name__ == "__main__":
    # Load the data
    data = load_customer_data()
    print("\n✨ Commit 1 complete: Data loading pipeline initialized!")