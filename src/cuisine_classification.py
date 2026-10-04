import matplotlib
matplotlib.use("Agg")
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/zomato_cleaned.csv")

print("Dataset Shape:", df.shape)
df["Primary Cuisine"] = (
    df["Cuisines"]
    .fillna("Unknown")
    .str.split(",")
    .str[0]
    .str.strip()
)
top_cuisines = (
    df["Primary Cuisine"]
    .value_counts()
    .head(10)
    .index
)
classification_df = df[
    df["Primary Cuisine"].isin(top_cuisines)
].copy()

features = [
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Has Table booking",
    "Has Online delivery"
]

X = classification_df[features]
y = classification_df["Primary Cuisine"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

lr_classifier = LogisticRegression(
    max_iter=3000,
    random_state=42
)

lr_classifier.fit(X_train, y_train)

lr_pred = lr_classifier.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=rf_classifier.classes_
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=rf_classifier.classes_,
    yticklabels=rf_classifier.classes_
)

plt.xlabel("Predicted Cuisine")
plt.ylabel("Actual Cuisine")
plt.title("Cuisine Classification")

os.makedirs("reports", exist_ok=True)

plt.tight_layout()

plt.savefig(
    "reports/cuisine_confusion_matrix.png",
    dpi=300
)

plt.close()

os.makedirs("models", exist_ok=True)

joblib.dump(
    rf_classifier,
    "models/cuisine_classifier.pkl"
)

print("\nConfusion matrix saved successfully!")
print("Cuisine model saved successfully!")

