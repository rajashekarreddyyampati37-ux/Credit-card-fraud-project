import pandas as pd
import numpy as np
import os

def load_data(filepath='data/dataset.csv'):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({'Time': np.random.uniform(0, 100000, n), 'Amount': np.random.exponential(50, n), 'V1': np.random.normal(0, 1, n), 'V2': np.random.normal(0, 1, n), 'Class': np.random.choice([0, 1], size=n, p=[0.95, 0.05])})
