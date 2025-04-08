
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def preprocess_data():
    # Load data
    data = pd.read_csv(r"C:\Projects\PB\data\water_potability.csv")
    
    # Add TDS estimate (Simple Feature Engineering)
    data['TDS_estimate'] = data['Solids'] + data['Chloramines']
    
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Split features & target
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for future use
    scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
    scaler_path.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    return X_train, X_test, y_train, y_test