import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap/Datasets/DietTypesDataset.csv")

# Encode categorical labels
le_diet = LabelEncoder()
df['Diet_type'] = le_diet.fit_transform(df['Diet_type'])
joblib.dump(le_diet, "CalorieMap/encoders/le_diet.pkl")

# Features and target
X = df[['Calories']].values
y = df[['Diet_type']].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_d = Input(shape=(1,))
x = Dense(64, activation='relu')(input_d)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output_d = Dense(1, activation='linear')(x)  # predicting encoded diet type

diet_model = Model(inputs=input_d, outputs=output_d)
diet_model.compile(optimizer='adam', loss='mse')

# Train model
diet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# Save model
diet_model.save("CalorieMap/models/diet_type_model.h5")
print("Diet type model trained and saved!")
