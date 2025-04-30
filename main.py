import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import xgboost as xgb
import os

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path, sheet_name="AAP_2022_city_v9")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    print("Data Information:")
    data.info()

    print("\nDescriptive Statistics:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    useless_cols = ['Status']
    data = data.drop(columns=useless_cols, errors='ignore')

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)
        # print(f"Column {col} filled with mean value: {mean_value:.4f}")
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna("Unknown")

    if 'City or Locality' in data.columns:
        data['City or Locality'] = data['City or Locality'].str.strip().str.lower()

    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()

    return data


def preprocess_data(data: pd.DataFrame):
    X = data[["PM10 (μg/m3)", "NO2 (μg/m3)", "PM10 temporal coverage (%)", "NO2 temporal coverage (%)", "PM25 temporal coverage (%)"]]
    y = data["PM2.5 (μg/m3)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def build_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nXGBoost Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R2 Score: {r2:.4f}")

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'feature_importance.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual PM2.5')
    plt.ylabel('Predicted PM2.5')
    plt.title('Actual vs Predicted PM2.5 Levels')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'xgboost_actual_vs_predicted.png'))
    plt.close()

    return model


def main():
    file_path = "data/data.xlsx"
    
    data = load_and_preprocess_data(file_path)
    if data is None:
        return

    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    model = build_and_evaluate_xgboost(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
