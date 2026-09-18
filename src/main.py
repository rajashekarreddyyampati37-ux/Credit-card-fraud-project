import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data
from src.model import FraudDetectionModel

def main():
    print("=========================================================")
    print("   Credit Card Fraud Detection CLI Tool")
    print("=========================================================\n")
    df = load_data('data/dataset.csv')
    print(f"[+] Loaded dataset with {len(df)} transaction records.")
    print("[+] Training Random Forest Classifier model...\n")
    print("--- Model Evaluation Results ---")
    model = FraudDetectionModel()
    model.train_and_evaluate(df)
    print("--- Manual Transaction Testing ---")
    while True:
        try:
            t_input = input("Enter Transaction Time (seconds, e.g., 500) or 'exit': ")
            if t_input.lower() == 'exit': break
            t = float(t_input)
            amt = float(input("Enter Transaction Amount ($): "))
            v1 = float(input("Enter Transaction Value V1 (e.g., -1.2): "))
            v2 = float(input("Enter Transaction Value V2 (e.g., 0.8): "))
            result = model.predict_single(t, amt, v1, v2)
            print(f"\n>>> Prediction Result: Transaction is [{result}]\n")
        except ValueError:
            print("[!] Invalid input. Please enter numbers.\n")

if __name__ == "__main__":
    main()
