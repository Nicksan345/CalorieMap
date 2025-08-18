# 🍽️ CalorieMap – AI-Powered Diet & Nutrition Prediction

CalorieMap is a machine learning-powered web application that helps predict **calories, diet type, recipe, cuisine type, and macronutrients (protein, carbs, fat)** from user inputs like calories and gender.  
It uses deep learning models trained on a custom dataset to provide **personalized nutrition insights**.

---

## 🚀 Features
- ✅ Predicts **calories** from user data  
- ✅ Suggests **diet type** (e.g., vegetarian, keto, etc.)  
- ✅ Predicts **recipe and cuisine type**  
- ✅ Estimates **macronutrients (Protein, Carbs, Fat)**  
- ✅ Simple **web interface** built with Flask  
- ✅ Models trained and saved as `.h5` for easy deployment  

---

## 🏗️ Project Structure
```plaintext
CalorieMap/
├── app.py                 # Main Flask application
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── models/                # Trained models
│   ├── calorie_model.h5
│   ├── diet_type_model.h5
│   ├── cuisine_model.h5
│   ├── recipe_model.h5
│   └── nutrition_model.h5
├── encoders/              # Encoders & scalers
│   ├── gender_encoder.pkl
│   ├── le_diet.pkl
│   ├── le_cuisine.pkl
│   ├── le_recipe.pkl
│   ├── scaler.pkl
│   └── scaler_nutrition.pkl
├── templates/             # HTML templates
│   ├── index.html
│   └── result.html
└── Datasets/              # Dataset files
│   └── DietTypesDataset.csv
|   └── CaloriePredictionDataset.csv
└── src/                   # Python scripts for training models
    ├── calorie_Prediction.py
    ├── Nutrition_Prediction.py
    ├── diet_type_Prediction.py
    ├── cuisine_Prediction.py
    └── recipe_Prediction.py
```


---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/CalorieMap.git
   cd CalorieMap
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the Flask app**
   ```bash
   python app.py

4. **Open in browser**
   ```bash
   http://127.0.0.1:5000/

## 📊 Models Used

- **Calorie Model** → Predicts calories from user features  
- **Diet Type Model** → Classifies diet type  
- **Recipe Model** → Suggests recipe  
- **Cuisine Model** → Predicts cuisine type  
- **Nutrition Model** → Estimates Protein, Carbs, and Fat

## 📦 Requirements

Dependencies are listed in `requirements.txt`.  
Install using:

```bash
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Create a Pull Request





   



