# Water Quality Potability Prediction

This project aims to predict whether water is **potable (safe to drink)** or **non-potable** using machine learning techniques based on various physicochemical properties of water.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Results](#results)
- [Challenges & Learnings](#challenges--learnings)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)

---

##  Overview

Access to clean drinking water is critical for health and well-being. This project helps identify unsafe water sources by predicting potability based on chemical parameters like pH, solids, chloramines, turbidity, etc.

---

##  Objectives

- Build a machine learning model to classify water as **potable** or **non-potable**.
- Analyze and understand the factors affecting water quality.
- Provide an interpretable solution to assist public health decisions.

---

##  Dataset

- **Samples**: 3,279
- **Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity, Potability
- **Target**: `Potability` (0 = Non-potable, 1 = Potable)

**Preprocessing:**
- Handled missing values via mean imputation.
- Engineered a new feature: `TDS_estimate = Solids + Chloramines`

---

##  Methodology

- **Data Preprocessing**: 
  - Handled missing values
  - Feature scaling using `StandardScaler`
  - Feature engineering (TDS_estimate)

- **Model Used**: Random Forest Classifier (RFC)
  - `n_estimators = 200`
  - `max_depth = 10`
  - `class_weight = 'balanced'`

- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix

---

## 📈 Model Performance

| Metric     | Score |
|------------|-------|
| Accuracy   | ~65%  |
| Precision  | 0.62  |
| Recall     | 0.65  |
| F1-Score   | 0.63  |

### Confusion Matrix

|              | Predicted: Non-Potable | Predicted: Potable |
|--------------|-------------------------|---------------------|
| Actual: Non-Potable | 320 (TN)              | 180 (FP)            |
| Actual: Potable     | 150 (FN)              | 250 (TP)            |

---

##  Feature Importance

Top 5 features impacting potability prediction:

1. **TDS_estimate** (engineered)
2. **Solids**
3. **Chloramines**
4. **pH**
5. **Turbidity**

---

##  Challenges & Learnings

### Challenges
- Class imbalance (only 39% potable samples)
- Correlated features (e.g., TDS and Solids)
- Balancing accuracy vs interpretability

### Key Learnings
- Feature engineering improved model performance
- Class weight balancing handled imbalance well
- Confusion matrix gave better insights than accuracy alone

---

##  Conclusion

- Achieved ~65% accuracy in classifying water safety
- Identified major contributing factors to water potability
- Model can support early intervention for unsafe water detection

---

## 🚀 Future Work

- Try more interpretable models like Logistic Regression with feature selection
- Explore oversampling (SMOTE) or ensemble methods
- Deploy as a web app for real-time water quality checks

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for EDA)
- Jupyter Notebook / VSCode

---

##  Folder Structure (Optional)

```bash
├── data/              # Raw or sample data files
├── models/            # Saved model files (if any)
├── notebooks/         # Jupyter notebooks for EDA & modeling
├── src/               # Python scripts (if separated)
├── report/            # Project report in .docx or .pdf
├── README.md
└── requirements.txt   # Python dependencies
