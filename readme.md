# Customer Retention & CLV Optimization System
*Predictive Modeling and Messaging Simulation for High-Value Customer Retention*

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview
A machine learning-driven system designed to halt declining retention in mid- and high-value customer segments. It features churn prediction, CLV estimation, and pre-campaign simulations of personalized messaging strategies to inform targeted, ROI-maximizing outreach.

📘 For a detailed breakdown of the problem definition, scoping process, stakeholder requirements, and full project methodology, see the [Methodology Document](docs/methodology.md)

---

## 📊 Business Impact
- **22% projected uplift** in customer lifetime value  
- **30% increase** in repeat purchases among at-risk segments  
- **3× ROI** expected from targeted marketing campaigns  
- Clear, early **churn signals** enable timely intervention

---

## 🔧 Technical Stack
- **Languages**: Python 3.8+, SQL  
- **ML Libraries**: scikit-learn, XGBoost  
- **Data Processing**: pandas, numpy  
- **Simulation & Uplift Modeling**  
- **Visualization**: matplotlib, seaborn, plotly  
- **Workflow Automation**: Python data pipelines  
- **Testing**: pytest  

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Customer transaction & engagement logs
SQL access or flat-file data sources
```

### Installation
```bash
# Clone the repo
git clone https://github.com/nihemelandu/customer-lifetime-value.git
cd customer-lifetime-value

<!--
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
-->

# Run test suite
pytest tests/

# Verify setup
python -c "import src; print('Setup OK')"
```

---

## 🔍 Key Features

### 1. Churn Prediction  
- **Models**: XGBoost, Random Forest  
- **Features**: Order recency/frequency, inactivity periods, discount use  
- **Evaluation**: ROC AUC, precision/recall, lift  
- **Output**: High-risk customer segments

### 2. CLV Estimation  
- **Approach**: Regression-based CLV model  
- **Scope**: Focus on mid- and high-LTV segments  
- **Validation**: Error tracking vs historical spend  
- **Output**: Estimated lifetime value per customer

### 3. Personalized Messaging Simulation  
- **Method**: Uplift modeling with scenario testing  
- **Application**: Pre-launch campaign projection  
- **Output**: Estimated CLV uplift and ROI by segment

---

## 📈 Results

### Problem Resolution
- Detected early churn among high-value customers  
- Segmented intervention priorities by expected ROI  
- Forecast campaign outcomes to justify marketing spend

### Model Performance
- **Churn ROC AUC**: ~0.84, **Recall**: 75% for high-risk segments  
- **CLV Error Margin**: <15%  
- **Simulated Impact**: 22% CLV lift, 30% repeat purchase increase, 3× ROI

---

## 🗂️ Repository Structure

```
clv-retention-optimizer/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_churn_modeling.ipynb
│   ├── 03_clv_prediction.ipynb
│   ├── 04_messaging_simulation.ipynb
│   └── 05_results_summary.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── churn.py
│   ├── clv.py
│   ├── simulation.py
│   ├── evaluation.py
│   └── utils.py
├── tests/
│   ├── test_churn.py
│   ├── test_clv.py
│   ├── test_simulation.py
│   └── test_utils.py
├── results/
│   ├── figures/
│   ├── models/
│   └── reports/
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    └── technical_specs.md
```

---

## 🔬 Methodology

### Data Science Pipeline
1. EDA: Identify churn trends and high-value segments  
2. Feature Engineering: RFM, engagement drops, discount usage  
3. Supervised Learning: Build churn classification & CLV prediction models  
4. Simulation: Model messaging impact using uplift techniques  
5. Validation: Use holdout data to forecast ROI and segment response

### Approach to Problem Solving
- Target high-value customer retention proactively  
- Pre-test campaign impact to guide marketing investment  
- Prioritize segments based on uplift and ROI  
- Modular pipeline structure allows flexible deployment

---

## 📓 Usage Examples

### 🎯 Predict Churn & CLV

Run the churn prediction pipeline using different churn labeling methods:
```bash
  python churn-pipeline.py --method fixed_threshold
  python churn-pipeline.py --method interpurchase_analysis
```

### Simulate Messaging Impact
```python
from src.simulation import MessagingSimulator

sim = MessagingSimulator()
sim_results = sim.run(churn_scores, clv_scores)

print(sim_results.head())
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest --cov=src --cov-report=html
```

---

## 📊 Data Requirements
- **Transaction History**: Orders, dates, values  
- **Engagement Data**: App/email activity logs  
- **Campaign History (optional)**: Past messages & responses  
- **Customer Profiles (optional)**: Demographics, segments

*Note: Synthetic/data samples available in `data/synthetic/`.*

---

## 🔄 Maintenance & Monitoring
- **Retrain Models**: Quarterly or monthly refresh  
- **Check Performance**: Monitor model drift & ROI impact  
- **Revalidate Segments**: Ensure segment stability  
- **Business Recap**: Monthly stakeholder updates

---

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/xyz`)  
3. Commit changes (`git commit -m 'Add xyz'`)  
4. Push your branch (`git push origin feature/xyz`)  
5. Open a Pull Request

---

## 🏷️ Tags
`churn-prediction` `clv-modeling` `marketing-analytics` `uplift-modeling` `customer-retention` `python` `xgboost` `scikit-learn`
