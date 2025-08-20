import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomerSegment:
    """Data class for customer segments"""
    segment_id: str
    characteristics: Dict
    clv_range: Tuple[float, float]
    retention_risk: str
    sustainability_score: float

class DataProcessor:
    """Handle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_purchase_journey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from purchase journey data"""
        # Sort by customer and date
        df = df.sort_values(['customer_id', 'purchase_date'])
        
        # Calculate time-based features
        df['days_since_last_purchase'] = df.groupby('customer_id')['purchase_date'].diff().dt.days
        df['purchase_frequency'] = df.groupby('customer_id').cumcount() + 1
        
        # Rolling window features
        df['avg_purchase_amount_30d'] = df.groupby('customer_id')['purchase_amount'].rolling(window=30, min_periods=1).mean().reset_index(drop=True)
        df['purchase_trend'] = df.groupby('customer_id')['purchase_amount'].rolling(window=5, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).reset_index(drop=True)
        
        # Sustainability features
        df['eco_product_ratio'] = df.groupby('customer_id')['is_eco_product'].expanding().mean().reset_index(drop=True)
        df['sustainability_engagement'] = df.groupby('customer_id')['sustainability_score'].expanding().mean().reset_index(drop=True)
        
        # Seasonal features
        df['purchase_month'] = df['purchase_date'].dt.month
        df['purchase_quarter'] = df['purchase_date'].dt.quarter
        df['is_holiday_season'] = df['purchase_month'].isin([11, 12]).astype(int)
        
        return df
    
    def create_customer_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregate features"""
        agg_features = df.groupby('customer_id').agg({
            'purchase_amount': ['sum', 'mean', 'std', 'count'],
            'days_since_last_purchase': ['mean', 'std'],
            'is_eco_product': ['mean', 'sum'],
            'sustainability_score': ['mean', 'max'],
            'purchase_frequency': 'max',
            'purchase_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['customer_id'] + [f"{col[0]}_{col[1]}" for col in agg_features.columns[1:]]
        
        # Calculate customer lifetime
        agg_features['customer_lifetime_days'] = (agg_features['purchase_date_max'] - agg_features['purchase_date_min']).dt.days
        agg_features['recency_days'] = (datetime.now() - agg_features['purchase_date_max']).dt.days
        
        return agg_features
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling"""
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Handle missing values
        df = df.fillna(df.median())
        
        return df

class CLVPredictor:
    """Customer Lifetime Value prediction models"""
    
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.features = None
        self.is_trained = False
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train both XGBoost and Random Forest models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Random Forest model
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate models
        xgb_pred = self.xgb_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)
        
        results = {
            'xgb_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'xgb_feature_importance': dict(zip(X.columns, self.xgb_model.feature_importances_)),
            'rf_feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_))
        }
        
        self.features = X.columns.tolist()
        self.is_trained = True
        
        logger.info(f"XGBoost RMSE: {results['xgb_rmse']:.2f}")
        logger.info(f"Random Forest RMSE: {results['rf_rmse']:.2f}")
        
        return results
    
    def predict_clv(self, X: pd.DataFrame, model_type: str = 'ensemble') -> np.ndarray:
        """Predict CLV using specified model or ensemble"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_type == 'xgb':
            return self.xgb_model.predict(X)
        elif model_type == 'rf':
            return self.rf_model.predict(X)
        else:  # ensemble
            xgb_pred = self.xgb_model.predict(X)
            rf_pred = self.rf_model.predict(X)
            return (xgb_pred + rf_pred) / 2

class ChurnPredictor:
    """Predict customer churn and retention risk"""
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.is_trained = False
    
    def create_churn_labels(self, df: pd.DataFrame, days_threshold: int = 90) -> pd.Series:
        """Create churn labels based on recency"""
        return (df['recency_days'] > days_threshold).astype(int)
    
    def train_churn_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train churn prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        self.is_trained = True
        logger.info(f"Churn Model AUC: {results['auc_score']:.3f}")
        
        return results
    
    def predict_churn_risk(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict churn probability and risk level"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        churn_proba = self.model.predict_proba(X)[:, 1]
        risk_levels = np.where(churn_proba > 0.7, 'High',
                              np.where(churn_proba > 0.4, 'Medium', 'Low'))
        
        return churn_proba, risk_levels

class CustomerSegmentation:
    """Customer segmentation and personalization"""
    
    def __init__(self):
        self.segments = {}
    
    def create_segments(self, df: pd.DataFrame) -> Dict[str, CustomerSegment]:
        """Create customer segments based on CLV and sustainability"""
        segments = {}
        
        # Eco-conscious high-value customers
        eco_high = df[(df['clv_predicted'] > df['clv_predicted'].quantile(0.8)) & 
                     (df['sustainability_engagement'] > 0.6)]
        segments['eco_champions'] = CustomerSegment(
            segment_id='eco_champions',
            characteristics={'high_clv': True, 'eco_conscious': True},
            clv_range=(eco_high['clv_predicted'].min(), eco_high['clv_predicted'].max()),
            retention_risk='Low',
            sustainability_score=eco_high['sustainability_engagement'].mean()
        )
        
        # Price-sensitive eco customers
        eco_medium = df[(df['clv_predicted'].between(df['clv_predicted'].quantile(0.4), 
                                                    df['clv_predicted'].quantile(0.8))) & 
                       (df['sustainability_engagement'] > 0.4)]
        segments['eco_adopters'] = CustomerSegment(
            segment_id='eco_adopters',
            characteristics={'medium_clv': True, 'eco_interested': True},
            clv_range=(eco_medium['clv_predicted'].min(), eco_medium['clv_predicted'].max()),
            retention_risk='Medium',
            sustainability_score=eco_medium['sustainability_engagement'].mean()
        )
        
        # Traditional high-value customers
        traditional_high = df[(df['clv_predicted'] > df['clv_predicted'].quantile(0.8)) & 
                             (df['sustainability_engagement'] <= 0.4)]
        segments['traditional_premium'] = CustomerSegment(
            segment_id='traditional_premium',
            characteristics={'high_clv': True, 'traditional': True},
            clv_range=(traditional_high['clv_predicted'].min(), traditional_high['clv_predicted'].max()),
            retention_risk='Low',
            sustainability_score=traditional_high['sustainability_engagement'].mean()
        )
        
        self.segments = segments
        return segments
    
    def assign_segments(self, df: pd.DataFrame) -> pd.Series:
        """Assign customers to segments"""
        segments = []
        
        for _, row in df.iterrows():
            if row['clv_predicted'] > df['clv_predicted'].quantile(0.8):
                if row['sustainability_engagement'] > 0.6:
                    segments.append('eco_champions')
                else:
                    segments.append('traditional_premium')
            elif row['clv_predicted'] > df['clv_predicted'].quantile(0.4):
                if row['sustainability_engagement'] > 0.4:
                    segments.append('eco_adopters')
                else:
                    segments.append('traditional_medium')
            else:
                segments.append('low_value')
        
        return pd.Series(segments, index=df.index)

class PersonalizedMessaging:
    """Generate personalized marketing messages"""
    
    def __init__(self):
        self.message_templates = {
            'eco_champions': {
                'subject': 'Exclusive Sustainable Collection Just for You',
                'content': 'As a sustainability leader, discover our latest eco-innovations that align with your values.',
                'cta': 'Shop Sustainable Collection',
                'offer': '15% off + free carbon-neutral shipping'
            },
            'eco_adopters': {
                'subject': 'Small Steps, Big Impact - Sustainable Choices',
                'content': 'Continue your sustainability journey with products that make a difference.',
                'cta': 'Explore Eco-Friendly Options',
                'offer': '10% off sustainable products'
            },
            'traditional_premium': {
                'subject': 'Premium Quality, Exclusive Access',
                'content': 'Experience our finest products crafted for discerning customers like you.',
                'cta': 'Shop Premium Collection',
                'offer': '20% off premium range'
            }
        }
    
    def generate_message(self, segment: str, customer_name: str = None) -> Dict:
        """Generate personalized message for customer segment"""
        if segment not in self.message_templates:
            segment = 'traditional_premium'  # Default fallback
        
        template = self.message_templates[segment]
        
        message = {
            'segment': segment,
            'subject': template['subject'],
            'content': template['content'],
            'cta': template['cta'],
            'offer': template['offer'],
            'personalized': bool(customer_name)
        }
        
        if customer_name:
            message['subject'] = f"{customer_name}, {template['subject']}"
        
        return message

class MarketingAutomation:
    """Automated marketing pipeline"""
    
    def __init__(self, clv_predictor: CLVPredictor, churn_predictor: ChurnPredictor,
                 segmentation: CustomerSegmentation, messaging: PersonalizedMessaging):
        self.clv_predictor = clv_predictor
        self.churn_predictor = churn_predictor
        self.segmentation = segmentation
        self.messaging = messaging
        self.campaign_results = []
    
    def run_campaign(self, customer_data: pd.DataFrame) -> Dict:
        """Run automated marketing campaign"""
        # Predict CLV
        clv_predictions = self.clv_predictor.predict_clv(customer_data)
        customer_data['clv_predicted'] = clv_predictions
        
        # Predict churn risk
        churn_prob, risk_levels = self.churn_predictor.predict_churn_risk(customer_data)
        customer_data['churn_probability'] = churn_prob
        customer_data['risk_level'] = risk_levels
        
        # Segment customers
        customer_data['segment'] = self.segmentation.assign_segments(customer_data)
        
        # Generate personalized messages
        messages = []
        for _, customer in customer_data.iterrows():
            message = self.messaging.generate_message(
                customer['segment'],
                customer.get('customer_name', None)
            )
            messages.append(message)
        
        # Campaign targeting logic
        high_value_eco = customer_data[
            (customer_data['segment'] == 'eco_champions') |
            ((customer_data['segment'] == 'eco_adopters') & (customer_data['churn_probability'] > 0.4))
        ]
        
        campaign_results = {
            'total_customers': len(customer_data),
            'targeted_customers': len(high_value_eco),
            'targeting_rate': len(high_value_eco) / len(customer_data),
            'expected_clv_uplift': self.calculate_expected_uplift(high_value_eco),
            'segment_distribution': customer_data['segment'].value_counts().to_dict(),
            'risk_distribution': customer_data['risk_level'].value_counts().to_dict()
        }
        
        self.campaign_results.append(campaign_results)
        
        return campaign_results
    
    def calculate_expected_uplift(self, targeted_customers: pd.DataFrame) -> Dict:
        """Calculate expected campaign uplift"""
        # Based on historical performance metrics
        eco_champions_uplift = 0.22  # 22% CLV uplift
        eco_adopters_uplift = 0.15   # 15% CLV uplift
        
        eco_champions = targeted_customers[targeted_customers['segment'] == 'eco_champions']
        eco_adopters = targeted_customers[targeted_customers['segment'] == 'eco_adopters']
        
        expected_uplift = {
            'eco_champions_uplift': len(eco_champions) * eco_champions['clv_predicted'].mean() * eco_champions_uplift,
            'eco_adopters_uplift': len(eco_adopters) * eco_adopters['clv_predicted'].mean() * eco_adopters_uplift,
            'total_uplift': (len(eco_champions) * eco_champions['clv_predicted'].mean() * eco_champions_uplift +
                           len(eco_adopters) * eco_adopters['clv_predicted'].mean() * eco_adopters_uplift)
        }
        
        return expected_uplift

class PipelineOrchestrator:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.clv_predictor = CLVPredictor()
        self.churn_predictor = ChurnPredictor()
        self.segmentation = CustomerSegmentation()
        self.messaging = PersonalizedMessaging()
        self.automation = None
    
    def train_pipeline(self, purchase_data: pd.DataFrame) -> Dict:
        """Train the complete pipeline"""
        logger.info("Starting pipeline training...")
        
        # Process purchase journey data
        journey_features = self.data_processor.create_purchase_journey_features(purchase_data)
        customer_features = self.data_processor.create_customer_aggregates(journey_features)
        
        # Prepare features for modeling
        model_features = self.data_processor.prepare_features(customer_features.copy())
        
        # Calculate actual CLV (or use historical CLV)
        actual_clv = model_features['purchase_amount_sum']  # Simplified CLV calculation
        
        # Train CLV prediction models
        feature_cols = [col for col in model_features.columns if col not in ['customer_id', 'purchase_amount_sum']]
        clv_results = self.clv_predictor.train_models(model_features[feature_cols], actual_clv)
        
        # Train churn prediction model
        churn_labels = self.churn_predictor.create_churn_labels(model_features)
        churn_results = self.churn_predictor.train_churn_model(model_features[feature_cols], churn_labels)
        
        # Set up automation
        self.automation = MarketingAutomation(
            self.clv_predictor, self.churn_predictor,
            self.segmentation, self.messaging
        )
        
        results = {
            'clv_model_performance': clv_results,
            'churn_model_performance': churn_results,
            'features_used': feature_cols,
            'training_samples': len(model_features)
        }
        
        logger.info("Pipeline training completed successfully!")
        return results
    
    def run_prediction_pipeline(self, new_data: pd.DataFrame) -> Dict:
        """Run prediction pipeline on new data"""
        if self.automation is None:
            raise ValueError("Pipeline must be trained before running predictions")
        
        # Process new data
        journey_features = self.data_processor.create_purchase_journey_features(new_data)
        customer_features = self.data_processor.create_customer_aggregates(journey_features)
        model_features = self.data_processor.prepare_features(customer_features.copy())
        
        # Run automated campaign
        campaign_results = self.automation.run_campaign(model_features)
        
        return campaign_results
    
    def save_pipeline(self, filepath: str):
        """Save trained pipeline to file"""
        pipeline_data = {
            'data_processor': self.data_processor,
            'clv_predictor': self.clv_predictor,
            'churn_predictor': self.churn_predictor,
            'segmentation': self.segmentation,
            'messaging': self.messaging
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load trained pipeline from file"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.data_processor = pipeline_data['data_processor']
        self.clv_predictor = pipeline_data['clv_predictor']
        self.churn_predictor = pipeline_data['churn_predictor']
        self.segmentation = pipeline_data['segmentation']
        self.messaging = pipeline_data['messaging']
        
        self.automation = MarketingAutomation(
            self.clv_predictor, self.churn_predictor,
            self.segmentation, self.messaging
        )
        
        logger.info(f"Pipeline loaded from {filepath}")

# Example usage
def main():
    """Example usage of the CLV prediction framework"""
    
    # Initialize pipeline
    pipeline = PipelineOrchestrator()
    
    # Sample data creation (replace with your actual data)
    sample_data = pd.DataFrame({
        'customer_id': range(1000),
        'purchase_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'purchase_amount': np.random.gamma(2, 50, 1000),
        'is_eco_product': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
        'sustainability_score': np.random.beta(2, 5, 1000),
        'customer_name': [f'Customer_{i}' for i in range(1000)]
    })
    
    # Train pipeline
    training_results = pipeline.train_pipeline(sample_data)
    print("Training Results:", training_results)
    
    # Run prediction on new data
    new_data = sample_data.sample(100)  # Simulate new customer data
    campaign_results = pipeline.run_prediction_pipeline(new_data)
    print("Campaign Results:", campaign_results)
    
    # Save pipeline
    pipeline.save_pipeline('clv_pipeline.pkl')

if __name__ == "__main__":
    main()