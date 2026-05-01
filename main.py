import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Dataset
df = pd.read_csv("data/housing.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# 2. Data Cleaning
# Drop missing values
df = df.dropna()

# Convert categorical column to numeric
# (furnishingstatus: furnished, semi-furnished, unfurnished)
df = pd.get_dummies(df, drop_first=True)

# 3. Data Visualization
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Feature Selection
# Adjust based on available columns
X = df.drop("price", axis=1)   # all features except price
y = df["price"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R² Score:", r2)

# 9. Coefficients
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nFeature Importance:\n", coeff_df)

# 10. Save Model
joblib.dump(model, "model/house_price_model.pkl")
print("\nModel saved successfully!")

# 11. Example Prediction
# IMPORTANT: Match number of features exactly

sample = X_test.iloc[0:1]   # using real sample from test data
prediction = model.predict(sample)

print("\nExample Prediction:")
print("Actual Price:", y_test.iloc[0])
print("Predicted Price:", prediction[0])