import pandas as pd
df = pd.read_csv("data/zomato_cleaned.csv")

print("Dataset Shape:", df.shape)

features = [
    "Average Cost for two",
    "Price range",
    "Votes",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

target = "Aggregate rating"

# Create input (X) and output (y)
X = df[features]
y = df[target]

print("\nSelected Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# Feature Importance
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# -----------------------------------------
# Random Forest without Votes
# -----------------------------------------

# Remove Votes from features
features_without_votes = [
    "Average Cost for two",
    "Price range",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

X_no_votes = df[features_without_votes]
y_no_votes = df[target]

# Train-test split
X_train_nv, X_test_nv, y_train_nv, y_test_nv = train_test_split(
    X_no_votes,
    y_no_votes,
    test_size=0.2,
    random_state=42
)

# Create model
rf_no_votes = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
rf_no_votes.fit(X_train_nv, y_train_nv)

# Predict
y_pred_nv = rf_no_votes.predict(X_test_nv)

# Evaluate
mae_nv = mean_absolute_error(y_test_nv, y_pred_nv)
mse_nv = mean_squared_error(y_test_nv, y_pred_nv)
rmse_nv = np.sqrt(mse_nv)
r2_nv = r2_score(y_test_nv, y_pred_nv)

print("\nRandom Forest WITHOUT Votes:")
print("MAE:", mae_nv)
print("MSE:", mse_nv)
print("RMSE:", rmse_nv)
print("R2 Score:", r2_nv)

import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the Random Forest model
joblib.dump(rf_model, "models/rating_model.pkl")

print("\nRating model saved successfully!")
print("File: models/rating_model.pkl")
