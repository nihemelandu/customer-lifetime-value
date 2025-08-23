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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

class ChurnPredictionPipeline:
    # def __init__(self):
    #     self.raw_data = None
    #     self.clean_data = None
    #     self.customer_features = None
    #     self.model = None
    #     self.scaler = None
        
    # COMMIT 1: Initialize customer churn pipeline with data loading
    def load_customer_data(self, file_path='data/synthetic/customer_analytics_master.csv'):
        """
        Load customer analytics data from CSV file.
        Returns pandas DataFrame with basic info printed.
        """
        print("🔄 Loading customer analytics data...")
        try:
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

        except FileNotFoundError:
            print(f"❌ Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
        
        return df
    
    # COMMIT 2: Add basic data validation and cleaning steps
    def validate_data_quality(self, df):
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
    
    def clean_data(self, df):
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
            null_orders = cleaned_df.order_value.isnull().sum()
            if null_orders > 0:
                print("removed null orders")
                print(cleaned_df.shape)
                cleaned_df = cleaned_df[~cleaned_df.order_value.isnull()]
                print(cleaned_df.shape)
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
    
    def create_customer_features(self, df):
        """
        Create customer-level features for churn prediction.
        Aggregates transaction data into customer behavioral metrics.
        """
        print("\n🔧 Creating customer features...")
        
        # Convert registration_date columns to datetime
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        
        # Calculate analysis date (most recent order date)
        analysis_date = df['order_date'].max()
        print(f"   • Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
        
        # Group by customer to create features
        customer_features = df.groupby('customer_id').agg({
            # Basic customer info
            'registration_date': 'first',
            #'customer_segment': 'first',
            'acquisition_channel': 'first',
            'account_status': 'first',
            
            # Order behavior - RFM metrics
            'order_date': ['max', 'count'],  # Most recent order, frequency
            'order_value': ['sum', 'mean', 'std'],  # Monetary metrics
            'quantity': 'sum',
            
            # Product behavior
            'category': 'nunique',  # Product diversity
            'unit_price': 'mean',
            
            # Promotion usage
            'discount_percentage': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip('_') for col in customer_features.columns]
    
        # Rename columns to meaningful names
        customer_features = customer_features.rename(columns={
            'customer_id_': 'customer_id',
            'order_date_count': 'order_frequency',
            'order_value_sum': 'total_revenue',
            'order_value_mean': 'avg_order_value',
            'order_date_max': 'last_order_date',
            'registration_date_first': 'registration_date',
            #'customer_segment_first': 'customer_segment',
            'acquisition_channel_first': 'acquisition_channel',
            'account_status_first': 'account_status',
            'category_nunique': 'product_diversity',     
        })
        
        #RFM is a behavioral segmentation framework.
        #CLV is a financial metric.
        #RFM can be used as input features to estimate or model CLV, but RFM ≠ CLV
        
        # Calculate RFM features
        customer_features['days_since_last_order'] = (analysis_date - customer_features['last_order_date']).dt.days
        customer_features['customer_tenure_days'] = (analysis_date - customer_features['registration_date']).dt.days
        #customer_features['order_frequency'] = customer_features['total_orders'] / (customer_features['customer_tenure_days'] / 30)  # orders per month
        
        # Create risk indicators
        customer_features['high_value_customer'] = (customer_features['total_revenue'] > customer_features['total_revenue'].quantile(0.8)).astype(int)
        customer_features['recent_customer'] = (customer_features['customer_tenure_days'] <= 90).astype(int)
        customer_features['at_risk_recency'] = (customer_features['days_since_last_order'] > 60).astype(int)
        customer_features['low_frequency'] = (customer_features['order_frequency'] < 1).astype(int)  # less than 1 order per month
        
        # Product engagement
        #customer_features['product_diversity'] = customer_features['category_nunique']
        customer_features['uses_promotions'] = (customer_features['discount_percentage_count'] > 0).astype(int)
        customer_features['avg_discount'] = customer_features['discount_percentage_mean'].fillna(0)
        
        # Select final feature set
        feature_columns = [
            'customer_id', 'acquisition_channel',
            'account_status', 'days_since_last_order', 'customer_tenure_days',
            'total_revenue', 'avg_order_value', 'order_frequency',
            'product_diversity', 'high_value_customer', 'recent_customer',
            'at_risk_recency', 'low_frequency', 'uses_promotions', 'avg_discount'
        ]
        
        final_features = customer_features[feature_columns].copy()
        
        print(f"   ✅ Features created for {len(final_features):,} customers")
        print(f"   📊 Feature set: {len(feature_columns)-1} features")  # -1 for customer_id
        print("   🎯 Key features: days_since_last_order, total_revenue, order_frequency, product_diversity")
        
        return final_features
    
    def create_churn_target(self, df):
        """
        Create churn target variable based on business rules.
        Define churn as customers who haven't ordered in 90+ days.
        """
        print("\n🎯 Creating churn target variable...")
        
        # Calculate analysis date (most recent order date)
        analysis_date = df['order_date'].max()
        
        # Group by customer to find last order date
        customer_last_order = df.groupby('customer_id')['order_date'].max().reset_index()
        customer_last_order['days_since_last_order'] = (analysis_date - customer_last_order['order_date']).dt.days
        
        # Define churn: customers inactive for 90+ days
        churn_threshold = 90
        customer_last_order['churned'] = (customer_last_order['days_since_last_order'] >= churn_threshold).astype(int)
        
        churn_rate = customer_last_order['churned'].mean()
        print(f"   • Churn threshold: {churn_threshold} days")
        print(f"   • Churn rate: {churn_rate:.1%}")
        
        return customer_last_order[['customer_id', 'churned']]
    
    def prepare_model_data(self, features_df, churn_df):
        """
        Prepare features and target for machine learning model.
        Handle categorical encoding and feature selection.
        """
        print("\n🔧 Preparing data for modeling...")
        
        # Merge features with churn target
        model_data = features_df.merge(churn_df, on='customer_id', how='inner')
        
        # Select numeric features for baseline model
        numeric_features = [
            'days_since_last_order', 'customer_tenure_days',
            'total_revenue', 'avg_order_value', 'order_frequency',
            'product_diversity', 'avg_discount'
        ]
        
        # Add binary features (already 0/1)
        binary_features = [
            'high_value_customer', 'recent_customer', 'at_risk_recency',
            'low_frequency', 'uses_promotions'
        ]
        
        # Encode categorical features
        categorical_features = ['acquisition_channel', 'account_status']
        
        # Simple label encoding for baseline model
        le = LabelEncoder()
        encoded_categoricals = []
        
        for cat_col in categorical_features:
            if cat_col in model_data.columns:
                encoded_col = f"{cat_col}_encoded"
                model_data[encoded_col] = le.fit_transform(model_data[cat_col].astype(str))
                encoded_categoricals.append(encoded_col)
    
        print(model_data.head())
        # Final feature set
        feature_columns = numeric_features + binary_features + encoded_categoricals
        X = model_data[feature_columns]
        y = model_data['churned']
        
        print(f"   • Features selected: {len(feature_columns)}")
        print(f"   • Samples: {len(model_data):,} customers")
        print(f"   • Target distribution: {(1-y.mean()):.1%} retained, {y.mean():.1%} churned")
        
        return X, y, feature_columns
    
    # Task: Add baseline model training and evaluation to complete end-to-end pipeline
    # Goal: Create working churn prediction model with performance metrics
    
    def train_baseline_model(self, X, y, feature_columns):
        """
        Train baseline logistic regression model for churn prediction.
        """
        
        print("\n🤖 Training baseline logistic regression model...")
        
        # Check for missing values
        #print(X.customer_id[X.days_since_last_order.isnull()])
        print(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Training set: {len(X_train):,} samples")
        print(f"   • Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("   ✅ Model trained successfully!")
        print(f"   📊 ROC-AUC Score: {roc_auc:.3f}")
        
        # Print classification report
        print("\n📈 Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        # Feature importance (coefficients)    
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\n🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            direction = "↑ Churn Risk" if row['coefficient'] > 0 else "↓ Churn Risk"
            print(f"   • {row['feature']}: {row['coefficient']:.3f} ({direction})")
        
        return model, scaler, roc_auc# Commit 4: Implement baseline logistic regression model
    
    # Main pipeline orchestrator
    def run_complete_pipeline(self):
        """Run the complete end-to-end churn prediction pipeline."""
        print("🚀 Starting Complete Churn Prediction Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        raw_data = self.load_customer_data()
        if raw_data is None:
            return None
        
        # Step 2: Validate data quality
        validation_report = self.validate_data_quality(raw_data)
        
        # Step 3: Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Step 4: Create features
        customer_features = self.create_customer_features(cleaned_data)
        
        # Step 5: Create target
        churn_target = self.create_churn_target(cleaned_data)
        
        # Step 6: Prepare for modeling
        X, y, feature_names = self.prepare_model_data(customer_features, churn_target)
        
        # Step 7: Train model
        model, scaler, roc_auc = self.train_baseline_model(X, y, feature_names)
        
        print("\n" + "=" * 60)
        print("✨ PIPELINE COMPLETE - END-TO-END CHURN PREDICTION READY!")
        print(f"🎯 Final Model Performance: ROC-AUC = {roc_auc:.3f}")
        print(f"📊 Pipeline processed {len(customer_features):,} customers")
        print(f"🔧 Using {len(feature_names)} behavioral features")
        print("📈 Ready for the next sprint - sprint 2!")
        
        results = {
            'model': model,
            'scaler': scaler,
            'features': customer_features,
            'performance': roc_auc,
            'feature_names': feature_names
        }
        
        return results

    
# Run the complete pipeline
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Run complete end-to-end process
    results = pipeline.run_complete_pipeline()
    
    if results:
        print(f"\n🎉 Success! Churn prediction model ready with {results['performance']:.3f} ROC-AUC")
    else:
        print("\n❌ Pipeline failed. Check data file path and format.")
    
    
    
    
    