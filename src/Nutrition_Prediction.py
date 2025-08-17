import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap/Datasets/DietTypesDataset.csv")

# Nutrition features and target
X_nutrition = df[['Calories']].values
y_nutrition = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].values

# Scale target
scaler_nutrition = StandardScaler()
y_nutrition = scaler_nutrition.fit_transform(y_nutrition)
joblib.dump(scaler_nutrition, "CalorieMap/encoders/scaler_nutrition.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_nutrition, y_nutrition, test_size=0.2, random_state=42)

# Build model
input_n = Input(shape=(1,))
x = Dense(64, activation='relu')(input_n)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output_n = Dense(3, activation='linear')(x)

nutrition_model = Model(inputs=input_n, outputs=output_n)
nutrition_model.compile(optimizer='adam', loss='mse')

# Train model
nutrition_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# Save model
nutrition_model.save("CalorieMap/models/nutrition_model.h5")
print("Nutrition model trained and saved!")
