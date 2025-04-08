
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import preprocess_data

def train_model():
    # Create models directory if missing
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Train model with class weights
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_estimators=200,      # Moderate improvement
        max_depth=10           # Moderate improvement
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0 (Undrinkable)', '1 (Drinkable)'])
    plt.yticks([0, 1], ['0 (Undrinkable)', '1 (Drinkable)'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
    plt.savefig(models_dir / "confusion_matrix.png")  # Save plot
    plt.show()
    
    # Save model
    model_path = models_dir / "rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()