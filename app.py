from flask import Flask, request, render_template
import numpy as np
import joblib
import tensorflow as tf

# ---------------- Initialize App ----------------
app = Flask(__name__)

# ---------------- Load Models ----------------
calorie_model = tf.keras.models.load_model("CalorieMap/models/calorie_model.h5", compile=False)
nutrition_model = tf.keras.models.load_model("CalorieMap/models/nutrition_model.h5", compile=False)
diet_type_model = tf.keras.models.load_model("CalorieMap/models/diet_type_model.h5", compile=False)
recipe_model = tf.keras.models.load_model("CalorieMap/models/recipe_model.h5", compile=False)
cuisine_model = tf.keras.models.load_model("CalorieMap/models/cuisine_model.h5", compile=False)


# ---------------- Load Encoders/Scalers ----------------
gender_encoder = joblib.load("CalorieMap/encoders/gender_encoder.pkl")
le_diet = joblib.load("CalorieMap/encoders/le_diet.pkl")
le_recipe = joblib.load("CalorieMap/encoders/le_recipe.pkl")
le_cuisine = joblib.load("CalorieMap/encoders/le_cuisine.pkl")
scaler = joblib.load("CalorieMap/encoders/scaler.pkl")
scaler_nutrition = joblib.load("CalorieMap/encoders/scaler_nutrition.pkl")

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form
        age = float(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        gender = request.form["gender"]
        bmi = float(request.form["bmi"])
        bmr = float(request.form["bmr"])
        activity_level = float(request.form["activity_level"])

        # Encode gender
        gender_encoded = gender_encoder.transform([gender])[0]

        # Prepare features for calorie prediction
        features = np.array([[age, weight, height, gender_encoded, bmi, bmr, activity_level]])
        features_scaled = scaler.transform(features)

        # Predict Calories
        calories = calorie_model.predict(features_scaled)[0][0]
        calories = max(0, calories)  # No negative calories

        # Predict Nutrition (Protein, Carbs, Fat)
        calories_scaled = scaler_nutrition.transform(np.array([[calories]]))
        nutrition_pred = nutrition_model.predict(calories_scaled)[0]

        protein, carbs, fat = np.maximum(nutrition_pred, 0)  # Clip negatives

        # Predict Diet Type
        diet_pred = diet_type_model.predict(np.array([[calories]]))
        diet_type = le_diet.inverse_transform([np.argmax(diet_pred)])[0]

        # Predict Recipe
        recipe_pred = recipe_model.predict(np.array([[calories]]))
        recipe_name = le_recipe.inverse_transform([np.argmax(recipe_pred)])[0]

        # Predict Cuisine
        cuisine_pred = cuisine_model.predict(np.array([[calories]]))
        cuisine_type = le_cuisine.inverse_transform([np.argmax(cuisine_pred)])[0]

        # Render results
        return render_template(
            "result.html",
            calories=round(calories, 2),
            protein=round(protein, 2),
            carbs=round(carbs, 2),
            fat=round(fat, 2),
            diet_type=diet_type,
            recipe_name=recipe_name,
            cuisine_type=cuisine_type
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
