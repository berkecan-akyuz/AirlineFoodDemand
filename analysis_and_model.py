
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import sys
import subprocess

# Ensure libraries are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import seaborn
except ImportError:
    install('seaborn')
    import seaborn as sns
try:
    import sklearn
except ImportError:
    install('scikit-learn')
    from sklearn.linear_model import LinearRegression

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    return pd.read_csv("synthetic_flight_data.csv")

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Drop flight_id as it's just an identifier
    numeric_df = df.drop(columns=['flight_id'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_heatmap.png")
    plt.close()
    
    # 2. Distributions
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['total_food_demand', 'flight_duration', 'passenger_count']):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig("eda_distributions.png")
    plt.close()
    
    # 3. Scatter Plots vs Target
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='flight_duration', y='total_food_demand', data=df, alpha=0.5)
    plt.title("Duration vs Food Demand")
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='passenger_count', y='total_food_demand', data=df, alpha=0.5)
    plt.title("Passengers vs Food Demand")
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x='is_international', y='total_food_demand', data=df)
    plt.title("International vs Food Demand")
    
    plt.tight_layout()
    plt.savefig("eda_scatter.png")
    plt.close()

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

def train_and_eval(df):
    print("Training Models...")
    
    # Prepare Data
    X = df.drop(columns=['flight_id', 'total_food_demand'])
    y = df['total_food_demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    metrics = {}
    
    # --- 1. Baseline Model (Mean Predictor) ---
    y_pred_base = np.full(shape=y_test.shape, fill_value=y_train.mean())
    metrics['Baseline'] = evaluate_model("Baseline", y_test, y_pred_base)
    
    # --- 2. Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics['Linear Regression'] = evaluate_model("Linear Regression", y_test, y_pred_lr)
    
    # --- 3. Random Forest Regressor ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics['Random Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf)
    
    # Save Metrics
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # --- Visualization: Predicted vs Actual (RF) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Food Demand")
    plt.ylabel("Predicted Food Demand")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("model_actual_vs_predicted.png")
    plt.close()
    
    # --- Visualization: Residuals (RF) ---
    residuals = y_test - y_pred_rf
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution (Random Forest)")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig("model_residuals.png")
    plt.close()
    
    # --- Visualization: Feature Importance (RF) ---
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=features[indices])
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("model_feature_importance.png")
    plt.close()
    
    # --- Cost Analysis (Bonus) ---
    # Cost = (OverPredictions * 5) + (UnderPredictions * 20)
    over_pred = np.maximum(0, y_pred_rf - y_test)
    under_pred = np.maximum(0, y_test - y_pred_rf)
    
    # We count instances or sum of units? Rule says: "Cost = (OverPredictions * $5) + (UnderPredictions * $20)"
    # Usually this implies per unit of error. 
    cost = (np.sum(over_pred) * 5) + (np.sum(under_pred) * 20)
    print(f"Estimated Business Cost (Test Set): ${cost:,.2f}")
    
    return metrics

if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
    metrics = train_and_eval(df)
    print("Analysis Complete. Metrics:")
    print(json.dumps(metrics, indent=2))
