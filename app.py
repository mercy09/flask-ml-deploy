
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("customer_segment_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return "Customer Segmentation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in expected order
        features = [
            data["total_spent"],
            data["delivery_delay"],
            data["actual_delivery_time"],
            data["recency_days"],
            data["frequency"],
            data["payment_type_encoded"]
        ]

        # Preprocess and predict
        scaled_input = scaler.transform([features])
        prediction = model.predict(scaled_input)[0]

        return jsonify({"customer_segment": int(prediction)})

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
