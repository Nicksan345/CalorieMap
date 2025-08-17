import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap/Datasets/DietTypesDataset.csv")

# Encode recipe names
le_recipe = LabelEncoder()
df['Recipe_name'] = le_recipe.fit_transform(df['Recipe_name'])
joblib.dump(le_recipe, "CalorieMap/encoders/le_recipe.pkl")

# Features and target
X = df[['Calories']].values
y = df[['Recipe_name']].values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_r = Input(shape=(1,))
x = Dense(64, activation='relu')(input_r)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output_r = Dense(1, activation='linear')(x)  # predicting encoded recipe

recipe_model = Model(inputs=input_r, outputs=output_r)
recipe_model.compile(optimizer='adam', loss='mse')

# Train model
recipe_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# Save model
recipe_model.save("CalorieMap/models/recipe_model.h5")
print("Recipe model trained and saved!")
