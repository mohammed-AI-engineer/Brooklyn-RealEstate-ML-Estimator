# Brooklyn Real Estate ML Estimator
### High-Precision Property Valuation Engine v4.5

An enterprise-grade Machine Learning system designed to estimate real estate values in Brooklyn, NY. This project leverages the power of XGBoost Regression to provide accurate market valuations based on historical federal transaction data.

## Key Highlights
- ML Architecture: Optimized XGBoost Regressor with carefully tuned hyperparameters (learning_rate=0.006).
- Luxury Dashboard: An interactive themed interface built with Streamlit (Cyber-Gold UI).
- Real Estate Analytics: Real-time data processing including Building Age calculation and Price/SqFt normalization.
- Production Ready: Clean code structure with separated preprocessing and inference pipelines.

## Performance Metrics
- Algorithm: Gradient Boosted Decision Trees (XGBoost)
- R2 Score: 0.8965 (Strong Variance Explanation)
- Mean Absolute Error (MAE): $444,304
- Market Data: 13,727 Property Transaction Vectors analyzed.

## Tech Stack
- Core: Python 3.9+
- ML Libraries: Scikit-Learn, XGBoost, Pandas, NumPy
- Visualization: Plotly and Streamlit
- Persistence: Joblib for Model Serialization

## How to Run
1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/Brooklyn-RealEstate-ML-Estimator.git

2. Install requirements:
   pip install -r requirements.txt

3. Launch the Engine:
   streamlit run dashboard.py

---
2026 Brooklyn Real Estate Intelligence | Engineered for Precision.