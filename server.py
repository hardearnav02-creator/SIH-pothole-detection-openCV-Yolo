from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Urban Tracking API Bridge is running."

@app.route('/api/detections', methods=['GET'])
def get_detections():
    csv_path = "detections.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['Latitude', 'Longitude'])
            return jsonify(df.to_dict(orient='records'))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, port=5000)