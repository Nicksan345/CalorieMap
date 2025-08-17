import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv("CalorieMap\Datasets\DietTypesDataset.csv")

# Encode categorical outputs
le_diet = LabelEncoder()
le_recipe = LabelEncoder()
le_cuisine = LabelEncoder()

df['Diet_type'] = le_diet.fit_transform(df['Diet_type'])
df['Recipe_name'] = le_recipe.fit_transform(df['Recipe_name'])
df['Cuisine_type'] = le_cuisine.fit_transform(df['Cuisine_type'])

joblib.dump(le_diet, "le_diet.pkl")
joblib.dump(le_recipe, "le_recipe.pkl")
joblib.dump(le_cuisine, "le_cuisine.pkl")

# ---------------- Nutrition Regression Model ----------------
X_nutrition = df[['Calories']].values
y_nutrition = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].values

X_train_nut, X_val_nut, y_train_nut, y_val_nut = train_test_split(X_nutrition, y_nutrition, test_size=0.2, random_state=42)

nutrition_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_nut.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3)  # Protein, Carbs, Fat
])

nutrition_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
nutrition_model.fit(X_train_nut, y_train_nut, epochs=50, batch_size=16, validation_data=(X_val_nut, y_val_nut))

nutrition_model.save("nutrition_model.h5")

# ---------------- Diet Classification Model ----------------
X_class = df[['Calories']].values
y_class = df[['Diet_type', 'Recipe_name', 'Cuisine_type']].values

# For simplicity, we predict each output separately
# Diet_type
X_train_diet, X_val_diet, y_train_diet, y_val_diet = train_test_split(X_class, y_class[:,0], test_size=0.2, random_state=42)
diet_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_diet.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_class[:,0])), activation='softmax')
])
diet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
diet_model.fit(X_train_diet, y_train_diet, epochs=50, batch_size=16, validation_data=(X_val_diet, y_val_diet))
diet_model.save("diet_type_model.h5")
print("✅ Diet type model trained and saved as diet_type_model.h5")

# Repeat similar process for Recipe_name and Cuisine_type if needed
