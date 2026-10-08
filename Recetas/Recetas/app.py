from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

app = Flask(__name__)
# Absolute path example
data = pd.read_csv("DBrecipes1.csv", encoding='latin-1')

data["ingredients_quantities"] = data["ingredients_quantities"].apply(ast.literal_eval)
data["ingredients_list"] = data["ingredients_list"].apply(ast.literal_eval)

def combine_ingredients(row):
    quantities = row["ingredients_quantities"]
    ingredients = row["ingredients_list"]
    combined = []

    # Une cantidad con ingrediente (maneja si tienen longitudes diferentes)
    for i in range(min(len(quantities), len(ingredients))):
        combined.append(f"{quantities[i]} de {ingredients[i]}")

    return combined

data["ingredients_combined"] = data.apply(combine_ingredients, axis=1)
data["ingredients_text_combined"] = data["ingredients_combined"].apply(lambda x: ", ".join(x))

data["ingredients_text_combined"]


# Preprocess Ingredients

data["ingredients_list"] = data["ingredients_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Asegurar que no haya NaN ni floats
data["ingredients_list"] = data["ingredients_list"].apply(
    lambda x: x if isinstance(x, list) else []
)
data["text_only"] = data["ingredients_list"].apply(
    lambda lst: " ".join(lst).lower()
)
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['text_only'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[["calories", "servings", "prep_time_hour", "prep_time_min"]])

# Combine Features
ingredient_weight = 3.0
X_combined = np.hstack([X_numerical * 0.5, X_ingredients.toarray() * ingredient_weight])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features, allergy_list):
    input_features_scaled =scaler.transform([input_features[:4]])
    input_ingredients_transformed = vectorizer.transform([input_features[4]])
    input_combined = np.hstack([input_features_scaled * 0.5, input_ingredients_transformed.toarray() * ingredient_weight])
    distances, indeces = knn.kneighbors(input_combined)
    recommendations =data.iloc[indeces[0]]

    if allergy_list:
        def contains_allergen(row):
            text = row["ingredients_text_combined"].lower()
            return any(allergen in text for allergen in allergy_list)
        recommendations = recommendations[~recommendations.apply(contains_allergen, axis=1)]

    return recommendations[["recipe_name", "calories", "servings", "prep_time_hour", "prep_time_min", "ingredients_text_combined", "image_url"]].head(5)

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text
    
@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    if request.method == 'POST':
        try:
            calories = float(request.form['calories']) 
            servings = int(request.form['servings'])
            prep_time_hour = int(request.form['prep_time_hour'])
            prep_time_min = int(request.form['prep_time_min'])
            ingredients = request.form['ingredients']
            allergies = request.form['allergies'].strip().lower()
            allergy_list = [a.strip() for a in allergies.split(",")] if allergies else []

            if calories < 0 or servings <= 0 or prep_time_hour < 0 or prep_time_min < 0:
                error_message = "Por favor, ingrese solo valores positivos en los campos numéricos."
                return render_template('index.html', recommendations=[], error_message=error_message)
            if not ingredients:
                error_message = "Debe ingresar al menos un ingrediente."
                return render_template('index.html', recommendations=[], error_message=error_message)
            input_features = [calories, servings, prep_time_hour, prep_time_min, ingredients]
            recommendations = recommend_recipes(input_features, allergy_list)
            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate, error_message = None)
        except ValueError:
             error_message = "Por favor, ingrese solo números válidos en los campos correspondientes."
             return render_template('index.html', recommendations=[], error_message=error_message)
    
    
    return render_template('index.html', recommendations=[], error_message=None)

          

     
if __name__ == '__main__':
    app.run(debug=True)
        