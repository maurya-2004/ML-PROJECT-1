<!-- 
pip install -r requirements.txt
pip install pipenv
pipenv install scikit-learn numpy pandas  # Creates Pipfile and Pipfile.lock
## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv\Scripts\activate    # Windows --> 



Water Quality Potability Prediction Project

Overview
    This project focuses on predicting water potability (drinkability) based on various water quality   parameters. The system uses machine learning to classify whether water is safe for human consumption  based on its chemical properties.

Key Features
    Data Preprocessing: Handles missing values and performs feature engineering

    Machine Learning Model: Uses Random Forest Classifier with balanced class weights

    Feature Scaling: StandardScaler for normalizing feature distributions

    Model Evaluation: Includes accuracy metrics, classification report, and confusion matrix visualization

    Persistent Storage: Saves trained model and scaler for future use

Project Structure
Copy
water-quality-potability/
├── models/                    # Saved models and scalers
│   ├── rf_model.pkl           # Trained Random Forest model
│   ├── scaler.pkl             # Feature scaler
│   └── confusion_matrix.png   # Visualization of model performance
├── notebooks/
│   └── water_potability_analysis.ipynb  # Data analysis notebook
├── src/
│   ├── train.py               # Model training script
│   └── preprocess.py          # Data preprocessing script
└── data/
    └── water_potability.csv   # Original dataset

Dataset
The dataset contains the following water quality parameters:
    pH value
    Hardness
    Solids (Total dissolved solids)
    Chloramines
    Sulfate
    Conductivity
    Organic_carbon
    Trihalomethanes
    Turbidity
    Potability (Target: 1 = Potable, 0 = Not Potable)

Model Performance
The Random Forest classifier achieves:
    Balanced accuracy considering class imbalance
    Detailed classification report (precision, recall, f1-score)
    Visual confusion matrix for performance evaluation

Usage
Run preprocessing and training:
 python src/train.py
The script will:
    Preprocess the data (handle missing values, add features, scale)
    Train the Random Forest model
    Evaluate performance
    Save the model and scaler to models/ directory

Requirements
    Python 3.x
    Scikit-learn
    Pandas
    Joblib
    Matplotlib