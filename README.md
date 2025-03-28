# Telco Customer Churn Prediction

## Overview
This repository contains a comprehensive analysis of Telco customer churn data, demonstrating skills in data cleaning, exploratory data analysis, predictive modeling, and actionable business insights generation. The project includes predictive models to identify factors contributing to customer churn and a video presentation explaining the methodology and findings.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Video Presentation](#video-presentation)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Description
This project analyzes a telecommunications company's customer data to identify factors that contribute to customer churn. Customer churn (customers discontinuing service) represents significant revenue loss and acquisition costs for telecom companies. By identifying key churn predictors, telecommunications companies can implement targeted retention strategies to reduce churn rates. The analysis focuses on customer demographics, service usage patterns, contract information, and payment methods to provide actionable insights for business decision-making.

## Technologies Used
- **Programming Languages**: Python
- **Data Analysis Libraries**: Pandas, NumPy
- **Visualization Tools**: Matplotlib, Seaborn, Plotly Express
- **Machine Learning Frameworks**: Scikit-learn, XGBoost, SMOTE (for imbalanced data)
- **Development Environment**: Google Colab
- **Version Control**: Git/GitHub

## Project Structure
```
telco-churn-prediction/
│
├── data/                           # Data files (raw and processed)
│   ├── raw/                        # Original Telco customer dataset
│   └── processed/                  # Cleaned and transformed data
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_data_cleaning.ipynb      # Initial data cleaning and preprocessing
│   ├── 02_exploratory_analysis.ipynb # EDA and feature engineering
│   ├── 03_modeling.ipynb           # Model training and evaluation
│   └── 04_business_insights.ipynb  # Business insights and recommendations
│
├── scripts/                        # Reusable scripts
│   ├── data_processing.py          # Functions for data processing
│   ├── feature_engineering.py      # Feature creation functions
│   ├── model_utils.py              # Model training utilities
│   └── visualization.py            # Visualization functions
│
├── models/                         # Trained models
│   ├── logistic_regression.pkl     # Logistic regression model
│   ├── svm_model.pkl               # SVM model
│   ├── xgboost_model.pkl           # XGBoost model
│   └── neural_network.pkl          # Neural network model
│
├── results/                        # Generated analysis
│   ├── figures/                    # Generated visualizations
│   └── tables/                     # Performance metrics tables
│
├── docs/                           # Documentation
│   ├── data_dictionary.md          # Data field descriptions
│   └── methodology.md              # Analysis methodology details
│
├── presentation/                   # Presentation materials
│   ├── telco_churn_slides.pdf      # Project presentation slides
│   └── demo_video.mp4              # Video walkthrough
│
├── requirements.txt                # Project dependencies
├── README.md                       # This file
└── LICENSE                         # License information
```

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
```

Set up the environment:
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The dataset contains information about Telco customers including:
- Demographics (gender, senior citizen status, partners, dependents)
- Account information (tenure, contract type, payment method, paperless billing)
- Services (phone, internet, streaming, security, tech support)
- Billing (monthly charges, total charges)
- Churn status

A full data dictionary is available in `docs/data_dictionary.md`.

## Usage

1. Navigate to the notebooks directory:
```bash
cd notebooks
```

2. Launch Jupyter Notebook or open in Google Colab:
```bash
jupyter notebook
# or upload to Google Colab
```

3. Open and run the notebooks in sequence:
   - `01_data_cleaning.ipynb` - Initial data preparation and handling missing values
   - `02_exploratory_analysis.ipynb` - Detailed EDA and feature engineering
   - `03_modeling.ipynb` - Model training, evaluation, and feature importance
   - `04_business_insights.ipynb` - Key insights and business recommendations

4. To use the trained models for prediction:
```python
import pickle

# Load the model
with open('models/xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Prepare your data (X_new)
# Make predictions
predictions = model.predict(X_new)
```

## Key Findings

Here are the main insights from the analysis:

1. **High Churn with Fiber Optic Internet**: Approximately 69% of customers who churned had fiber optic internet service, despite its supposedly superior technical quality. This suggests issues with either service quality, pricing strategy, or customer expectations.

2. **Payment Method Impact**: Electronic check payment method showed significantly higher churn rates (45-53%) compared to other payment methods across all internet service types. This could indicate potential issues with electronic payment processing or a correlation with month-to-month contracts.

3. **Contract Type Correlation**: Month-to-month contracts had drastically higher churn rates compared to one-year or two-year contracts. About 88.6% of churned customers were on month-to-month contracts.

4. **Early Tenure Vulnerability**: 50% of customers who left did so within the first 10 months of service, with a substantial drop in retention after the first 5 months. This highlights a critical early relationship period.

5. **Security and Support Features**: Customers without tech support (83.4%) and online security (similar percentage) were much more likely to churn, suggesting these value-added services significantly improve retention.

![Customer Tenure Distribution](https://github.com/SaburBello/Telco-Customer-Prediction/edit/main/Customer%20tenure%20distribution_telco.png)
![Payment Method and Internet Service Churn Rates](https://github.com/SaburBello/Telco-Customer-Prediction/edit/main/Payment%20method%20and%20Internet%20service_telco.png)

## Video Presentation

A video presentation explaining the project methodology, analysis process, and key findings is available [here](https://www.veed.io/view/d7b7f3f5-d0c6-40cc-9fa1-5872f62b6acb?panel=share&sharingWidget=true). The video provides a walkthrough of the project and demonstrates the practical applications of the insights gained.

## Future Work

Potential extensions for this project include:

- **Customer Segmentation Analysis**: Implement clustering algorithms to identify distinct customer segments with similar behaviors and tailor retention strategies for each group.
- **Price Sensitivity Analysis**: Investigate the relationship between price points and churn rates, especially for fiber optic customers, to optimize pricing strategies.
- **Retention Campaign Simulation**: Develop models to simulate the impact of different retention strategies on reducing churn and calculate potential ROI.
- **Real-time Churn Prediction**: Create a production-ready ML pipeline to score customers daily/weekly for churn risk.
- **Survival Analysis**: Implement time-to-event models to predict not just if, but when customers are likely to churn.
- **Development of an interactive dashboard**: Create a Streamlit or Dash application for business stakeholders to explore churn patterns.

## Contact

- **Name**: Sabur Bello
- **LinkedIn**: www.linkedin.com/in/sabur-bello
- **Email**: bellosabur@gmail.com
- **Portfolio**: https://bit.ly/sabur-bello-data-analyst

---

Feel free to reach out with any questions or suggestions for collaboration!
