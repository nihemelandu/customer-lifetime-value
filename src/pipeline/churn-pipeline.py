# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:20:10 2025

@author: ngozi
"""

# Commit 1: Initialize customer churn pipeline with data loading
# Task: Load the customer analytics CSV file and display basic dataset information
# Goal: Create a simple script that reads CSV, shows shape, columns, and sample data

import pandas as pd
import numpy as np

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

def validate_data_quality(df):
    """
    Perform basic data quality validation checks.
    Returns validation report and flags any major issues.
    """
    print("\n🔍 Running data quality validation...")
    
    validation_report = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    missing_pct = (missing_values / len(df)) * 100
    
    print("\n📋 Missing Values Check:")
    for col in missing_values[missing_values > 0].index:
        print(f"   • {col}: {missing_values[col]:,} ({missing_pct[col]:.1f}%)")
    
    if missing_values.sum() == 0:
        print("   ✅ No missing values found")
    
    validation_report['missing_values'] = missing_values.to_dict()
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print("\n🔄 Duplicate Rows Check:")
    print(f"   • Total duplicates: {duplicate_count:,}")
    
    if duplicate_count == 0:
        print("   ✅ No duplicate rows found")
    
    validation_report['duplicates'] = duplicate_count
    
    # Check data types
    print("\n📊 Data Types Check:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"   • Numeric columns: {len(numeric_cols)} - {numeric_cols[:3]}{'...' if len(numeric_cols) > 3 else ''}")
    print(f"   • Text columns: {len(text_cols)} - {text_cols[:3]}{'...' if len(text_cols) > 3 else ''}")
    
    validation_report['data_types'] = {
        'numeric': numeric_cols,
        'text': text_cols
    }
    
    return validation_report

def clean_data(df):
    """
    Apply basic data cleaning steps.
    Returns cleaned DataFrame.
    """
    print("\n🧹 Applying data cleaning steps...")
    
    cleaned_df = df.copy()
    initial_rows = len(cleaned_df)
    
    # Remove exact duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        print(f"   • Removed {duplicates_removed:,} duplicate rows")
    
    # Clean numeric columns - remove negative values where they don't make sense
    numeric_issues = 0
    
    if 'order_value' in cleaned_df.columns:
        negative_orders = (cleaned_df['order_value'] < 0).sum()
        if negative_orders > 0:
            cleaned_df = cleaned_df[cleaned_df['order_value'] >= 0]
            numeric_issues += negative_orders
            print(f"   • Removed {negative_orders:,} rows with negative order values")
    
    if 'quantity' in cleaned_df.columns:
        zero_quantity = (cleaned_df['quantity'] <= 0).sum()
        if zero_quantity > 0:
            cleaned_df = cleaned_df[cleaned_df['quantity'] > 0]
            numeric_issues += zero_quantity
            print(f"   • Removed {zero_quantity:,} rows with zero/negative quantity")
    
    if 'unit_price' in cleaned_df.columns:
        negative_prices = (cleaned_df['unit_price'] < 0).sum()
        if negative_prices > 0:
            cleaned_df = cleaned_df[cleaned_df['unit_price'] >= 0]
            numeric_issues += negative_prices
            print(f"   • Removed {negative_prices:,} rows with negative unit prices")
    
    # Basic outlier check - extreme values that are likely data errors
    if 'order_value' in cleaned_df.columns:
        q99 = cleaned_df['order_value'].quantile(0.99)
        extreme_orders = (cleaned_df['order_value'] > q99 * 10).sum()  # 10x the 99th percentile
        if extreme_orders > 0:
            cleaned_df = cleaned_df[cleaned_df['order_value'] <= q99 * 10]
            print(f"   • Removed {extreme_orders:,} rows with extreme order values (>${q99*10:,.0f})")
    
    final_rows = len(cleaned_df)
    total_removed = initial_rows - final_rows
    
    print(f"   ✅ Cleaning complete: {total_removed:,} rows removed ({total_removed/initial_rows:.1%})")
    print(f"   📊 Final dataset: {final_rows:,} rows")
    
    return cleaned_df


def main():
    # Load the data
    data = load_customer_data()

    # Validate data quality
    validation_report = validate_data_quality(data)
    
    # Clean data
    cleaned_df = clean_data(data)
    
    print("\n✨ Commit 2 complete: Data validation and cleaning pipeline ready!")
    print(f"📈 Ready for feature engineering with {len(cleaned_df):,} clean records")
    
    return cleaned_df, validation_report

if __name__ == "__main__":
    cleaned_data, report = main()
    
    
    
    
    