import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap/Datasets/DietTypesDataset.csv")

# Encode cuisine types
le_cuisine = LabelEncoder()
df['Cuisine_type'] = le_cuisine.fit_transform(df['Cuisine_type'])
joblib.dump(le_cuisine, "CalorieMap/encoders/le_cuisine.pkl")

# Features and target
X = df[['Calories']].values
y = df[['Cuisine_type']].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_c = Input(shape=(1,))
x = Dense(64, activation='relu')(input_c)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output_c = Dense(1, activation='linear')(x)  # predicting encoded cuisine

cuisine_model = Model(inputs=input_c, outputs=output_c)
cuisine_model.compile(optimizer='adam', loss='mse')

# Train model
cuisine_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# Save model
cuisine_model.save("CalorieMap/models/cuisine_model.h5")
print("Cuisine model trained and saved!")
