from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train_and_evaluate(self, df):
        X = df[['Time', 'Amount', 'V1', 'V2']]
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
        
    def predict_single(self, time, amount, v1, v2):
        pred = self.model.predict([[time, amount, v1, v2]])
        return "Fraudulent" if pred[0] == 1 else "Legitimate"
