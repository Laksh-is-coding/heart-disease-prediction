import pandas as pd

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/heart_disease_uci.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

# =========================
# CHECK MISSING VALUES
# =========================
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# =========================
# TARGET CONVERSION
# =========================
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

print("\nTarget distribution:")
print(df['target'].value_counts())

# =========================
# DROP USELESS COLUMNS
# =========================
df = df.drop(['id', 'dataset', 'num'], axis=1)

# Drop high-missing columns
df = df.drop(['ca', 'thal', 'slope'], axis=1)

print("\nColumns after dropping:")
print(df.columns)

# =========================
# HANDLE MISSING VALUES
# =========================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numerical with mean
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# =========================
# ENCODE CATEGORICAL DATA
# =========================
df = pd.get_dummies(df, drop_first=True)

print("\nData after encoding:")
print(df.head())

# =========================
# SPLIT FEATURES & TARGET
# =========================
X = df.drop('target', axis=1)
y = df['target']

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# =========================
# TRAIN-TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# MODEL TRAINING
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_lr), 2))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 2))

# =========================
# EVALUATION
# =========================
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# =========================
# SAVE MODEL
# =========================
import pickle

with open("models/model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\n✅ Model saved successfully!")

# =========================
# SAVE CLEANED DATA
# =========================
df.to_csv("data/cleaned_heart.csv", index=False)

print("\n✅ Preprocessing complete. Cleaned dataset saved.")