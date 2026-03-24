from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("models/model.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

# API route (JSON)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)

        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]

        prediction = model.predict(input_df)[0]
        result = "Disease" if prediction == 1 else "No Disease"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# FORM route (UI)
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        data = request.form.to_dict()

        for key in data:
            data[key] = float(data[key])

        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)

        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "Disease" if prediction == 1 else "No Disease"
        probability_value = round(probability * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}",
            probability_text=f"Risk Score: {probability_value}%",
            probability_value=probability_value
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)