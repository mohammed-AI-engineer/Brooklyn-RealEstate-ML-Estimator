import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Path Configuration
# Using relative paths for seamless GitHub integration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "brooklyn_sales_raw.csv")

def run_real_estate_pipeline():
    # 2. Data Loading
    if not os.path.exists(FILE_PATH):
        print(f"Error: Dataset not found at {FILE_PATH}")
        return
        
    df = pd.read_csv(FILE_PATH)

    # 3. Feature Engineering & Cleaning
    # Converting sale date to year and dropping original date column
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])
    df['SALE_YEAR'] = df['SALE DATE'].dt.year
    df = df.drop(columns=['SALE DATE'])
    
    # Removing outliers and unrealistic data points
    df = df[df['GROSS SQUARE FEET'] > 300]
    df = df[df['SALE PRICE'] > 100000]

    # Filtering by price per square foot to remove extreme noise
    df['price_per_sqft'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
    df = df[(df['price_per_sqft'] >= 100) & (df['price_per_sqft'] <= 2500)]
    df = df.drop(columns=['price_per_sqft'])

    # Calculating building age at the time of sale
    df['BUILDING_AGE'] = df['SALE_YEAR'] - df['YEAR BUILT']
    df.loc[df['BUILDING_AGE'] < 0, 'BUILDING_AGE'] = 0

    # Removing non-predictive high-cardinality columns
    noise_cols = ['ADDRESS', 'LOT']
    df = df.drop(columns=[c for c in noise_cols if c in df.columns], errors='ignore')

    # 4. Encoding & Null Handling
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(0)

    # 5. Model Training Setup
    X = df.drop(columns=['SALE PRICE'])
    
    # Applying Log Transformation to the target variable to stabilize variance
    y_log = np.log1p(df['SALE PRICE'])
    
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # 6. XGBoost Regressor Configuration
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.006,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.4,
        n_jobs=-1,
        random_state=42
    )

    print("Status: Training in progress...")
    model.fit(X_train, y_train_log)
    print("Status: Training completed successfully.")

    # 7. Predictions & Inverse Transformation
    preds_log = model.predict(X_test)
    predictions = np.expm1(preds_log)
    y_test_actual = np.expm1(y_test_log)

    # 8. Evaluation Metrics
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

    print("\n--- Model Performance Metrics ---")
    print(f"R2 Score (Variance Explained): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

    # Comparison Table for validation
    comparison = pd.DataFrame({'Actual': y_test_actual, 'Predicted': predictions})
    comparison['Difference'] = np.abs(comparison['Actual'] - comparison['Predicted'])
    
    print("\nSample Comparisons (Actual vs Predicted):")
    print(comparison.head(10))

if __name__ == "__main__":
    run_real_estate_pipeline()