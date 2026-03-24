import pandas as pd
import pickle
import os

df = pd.read_csv("data/cleaned_heart.csv")

X = df.drop('target', axis=1)

print("Current working directory:", os.getcwd())

file_path = os.path.join(os.getcwd(), "models", "columns.pkl")

with open(file_path, "wb") as f:
    pickle.dump(X.columns, f)

print("Saved at:", file_path)