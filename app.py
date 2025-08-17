from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- Load Scaler & Encoders ----------------
scaler = joblib.load("CalorieMap/models/scaler.pkl")
gender_encoder = joblib.load("CalorieMap/models/gender_encoder.pkl")

le_diet = joblib.load("CalorieMap/models/le_diet.pkl")
le_recipe = joblib.load("CalorieMap/models/le_recipe.pkl")
le_cuisine = joblib.load("CalorieMap/models/le_cuisine.pkl")

# ---------------- Load Models (compile=False avoids mse error) ----------------
calorie_model = load_model("CalorieMap/models/calorie_model.h5", compile=False)
nutrition_model = load_model("CalorieMap/models/nutrition_model.h5", compile=False)
diet_type_model = load_model("CalorieMap/models/diet_type_model.h5", compile=False)

# ---------------- Flask Route ----------------
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            age = float(request.form['age'])
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            BMI = float(request.form['BMI'])
            BMR = float(request.form['BMR'])
            activity_level = float(request.form['activity_level'])
            gender = request.form['gender']

            # Encode gender
            gender_val = 0 if gender.lower() == 'Male' else 1


            # Scale features
            X_input = scaler.transform([[age, weight, height, gender_val, BMI, BMR, activity_level]])

            # Predict calories
            predicted_calories = calorie_model.predict(X_input)[0][0]

            # Predict macros
            macros = nutrition_model.predict(np.array([[predicted_calories]]))[0]
            protein, carbs, fat = macros[0], macros[1], macros[2]

            # Predict diet type
            diet_class = diet_type_model.predict(np.array([[predicted_calories]]))
            diet_type = le_diet.inverse_transform([diet_class.argmax(axis=1)[0]])[0]

            return render_template("form.html", result={
                "calories": round(predicted_calories,2),
                "protein": round(protein,2),
                "carbs": round(carbs,2),
                "fat": round(fat,2),
                "diet_type": diet_type
            })

        except Exception as e:
            return render_template("form.html", error=str(e))

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
