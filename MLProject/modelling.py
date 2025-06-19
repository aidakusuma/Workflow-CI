import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load dan prepare data untuk modeling
    """
    print("Loading and preparing data...")
    
    # Load data dari CSV file
    csv_path = r"C:\Users\A.I\Documents\dicoding submission\MSML\Membangun_model\Data\weatherdata_preprocessing.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from: {csv_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        print("Please check the file path and ensure the file exists.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, None
    
    # Preprocessing
    if 'Location' in df.columns:
        le = LabelEncoder()
        df['Location_encoded'] = le.fit_transform(df['Location'])
    
    # Features and target - sesuaikan dengan nama kolom di CSV Anda
    feature_columns = ['Location_encoded', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']
    X = df[feature_columns]
    y = df['Temperature_C']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X_train, X_test, y_train, y_test, feature_columns

def train_model(model, model_name, X_train, X_test, y_train, y_test, feature_columns):
    """
    Train model dengan MLflow tracking
    """
    with mlflow.start_run(run_name=f"{model_name}_experiment"):
        
        # Enable autolog untuk scikit-learn
        mlflow.sklearn.autolog()
        
        print(f"\n--- Training {model_name} ---")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Log additional metrics manually
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        
        # Log parameters manually
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Log feature names
        for i, feature in enumerate(feature_columns):
            mlflow.log_param(f"feature_{i+1}", feature)
        
        # Create and log visualization
        plt.figure(figsize=(12, 5))
        
        # Actual vs Predicted plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Temperature')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
        
        # Log the plot
        mlflow.log_artifact(f'{model_name}_evaluation.png')
        plt.close()
        
        # Print results
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        return model

def main():
    """
    Main function untuk menjalankan eksperimen
    """
    print("=== MLflow Weather Prediction Experiment ===")
    
    # Set MLflow tracking URI ke local
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment name
    experiment_name = "Weather_Prediction_Experiment"
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_columns = load_and_prepare_data()
    
    # Define models (tanpa hyperparameter tuning)
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }
    
    # Train models
    trained_models = {}
    for model_name, model in models.items():
        trained_model = train_model(model, model_name, X_train, X_test, y_train, y_test, feature_columns)
        trained_models[model_name] = trained_model
    
    print("\n=== Experiment Completed ===")
    print("Jalankan 'mlflow ui' untuk melihat hasil eksperimen")
    print("Buka browser dan kunjungi: http://localhost:5000")

if __name__ == "__main__":
    main()