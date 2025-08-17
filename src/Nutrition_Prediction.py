import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# ---------------- Load Dataset ----------------
df = pd.read_csv("CalorieMap\Datasets\DietTypesDataset.csv")

# Features and Targets
X_nutrition = df[['Calories']].values
y_nutrition = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].values

# Scale calories (important for stable training)
scaler_nutrition = StandardScaler()
X_nutrition_scaled = scaler_nutrition.fit_transform(X_nutrition)

# Save the scaler for later use in app.py
joblib.dump(scaler_nutrition, "CalorieMap/encoders/scaler_nutrition.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_nutrition_scaled, y_nutrition, test_size=0.2, random_state=42
)

# ---------------- Build Nutrition Model ----------------
nutrition_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='relu')
  
])

nutrition_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
nutrition_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# Save trained model
nutrition_model.save("CalorieMap/models/nutrition_model.h5")

print("Nutrition model trained and saved as nutrition_model.h5")

