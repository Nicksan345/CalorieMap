import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap\Datasets\CaloriesPredictionDataset.csv")

# Separate features (X) and target (y)
X = df[['age', 'weight(kg)', 'height(m)', 'gender', 'BMI', 'BMR', 'activity_level']]
y = df['calories']

# Encode gender (if it's categorical: Male/Female)
if df['gender'].dtype == 'object':
    le_gender = LabelEncoder()
    X['gender'] = le_gender.fit_transform(X['gender'])
    joblib.dump(le_gender, "gender_encoder.pkl")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build neural network model
calorie_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])

calorie_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
calorie_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Save the trained model
calorie_model.save("calorie_model.h5")

print("✅ Calorie model trained and saved as calorie_model.h5")
