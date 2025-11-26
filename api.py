from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS 

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)
# Mengizinkan akses dari domain frontend (Vercel)
CORS(app) 

# Load model path
port = int(os.environ.get('PORT', 5000))
# Path model harus relatif terhadap root folder Render
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# adjust this depending on where the file actually is in the repo:
# 1) if the file is in diacheckv2-back/model/rf_diabetes_smote.pkl 
#    and api.py is in diacheckv2-back/, use "model"
# 2) if the file is right beside api.py, remove "model" in the join.
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_diabetes_smote.pkl")

@app.route('/')
def home():
    """Endpoint untuk pengecekan status server."""
    return f"DiaCheck API is running. Model status: {'Ready' if model else 'Error'}", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi risiko."""
    if not model:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        data = request.get_json()
        
        # Helper function untuk konversi data yang aman
        def get_data(key, default):
            return float(data.get(key, default))
        
        # 1. Perhitungan BMI
        weight = get_data('weight', 0)
        height = get_data('height', 100) / 100 # cm ke m
        bmi = weight / (height * height) if height > 0 else 25.0

        # =================================================================
        # CRITICAL MAPPING: 21 Features dalam URUTAN YANG TEPAT
        # (Sesuai dengan feature_order pada training data Anda)
        # =================================================================
        
        # Fitur yang tidak ditanyakan (Didefault ke nilai paling aman/rata-rata)
        # ------------------------------------------------------------------
        CHOLCHECK = 1.0     # Asumsi: Sudah cek kolesterol
        STROKE_PROXY = get_data('HeartAttackOrStroke', 0.0) # Proxied dari pertanyaan Jantung/Stroke
        HEART_PROXY = get_data('HeartAttackOrStroke', 0.0)  # Proxied dari pertanyaan Jantung/Stroke
        FRUITS = 1.0        # Asumsi: Konsumsi buah
        VEGGIES = 1.0       # Asumsi: Konsumsi sayur
        ALCOHOL = 0.0       # Asumsi: Tidak konsumsi alkohol berat
        HEALTHCARE = 1.0    # Asumsi: Ada akses kesehatan
        NODOCCOST = 0.0     # Asumsi: Tidak ada kendala biaya
        MENTHLTH = 2.0      # Rata-rata 2 hari mental terganggu (Def. Aman)
        PHYSHLTH = 2.0      # Rata-rata 2 hari fisik terganggu (Def. Aman)

        features = [
            get_data('HighBP', 0.0),            # 1. HighBP
            get_data('HighChol', 0.0),          # 2. HighChol
            CHOLCHECK,                          # 3. CholCheck
            bmi,                                # 4. BMI (Calculated)
            get_data('Smoker', 0.0),            # 5. Smoker
            STROKE_PROXY,                       # 6. Stroke
            HEART_PROXY,                        # 7. HeartDiseaseorAttack
            get_data('PhysActivity', 0.0),      # 8. PhysActivity
            FRUITS,                             # 9. Fruits
            VEGGIES,                            # 10. Veggies
            ALCOHOL,                            # 11. HvyAlcoholConsump
            HEALTHCARE,                         # 12. AnyHealthcare
            get_data('NoDoc', NODOCCOST), # 13. NoDocbcCost (Mengambil 0.0 dari def)
            get_data('GenHlth', 3.0),           # 14. GenHlth
            MENTHLTH,                           # 15. MentHlth
            PHYSHLTH,                           # 16. PhysHlth
            get_data('DiffWalk', 0.0),          # 17. DiffWalk
            get_data('Sex', 0.0),               # 18. Sex
            get_data('age', 1.0),               # 19. Age
            get_data('Education', 4.0),         # 20. Education (Input dinamis)
            get_data('Income', 5.0)             # 21. Income (Input dinamis)
        ]

        try:
            model_bundle = joblib.load(MODEL_PATH)
            imputer = model_bundle["imputer"]
            model = model_bundle["model"]

            X = np.array([features])
            X_imp = imputer.transform(X)
            probability = model.predict_proba(X_imp)

            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print("Error loading model:", e)
            model = None
        
        # Risk Score = P(Prediabetes) + P(Diabetes)
        # Class 0: No Diabetes, Class 1: Prediabetes, Class 2: Diabetes
        risk_score = probability[0][1] + probability[0][2] 

        return jsonify({
            'risk_score': float(risk_score),
            'status': 'success'
        })

    except Exception as e:
        # Memberikan error yang jelas jika terjadi kegagalan data/pemrosesan
        return jsonify({'error': f'Processing or data format error: {str(e)}'}), 400

if __name__ == '__main__':
    # Untuk testing lokal
    app.run(host='0.0.0.0', port=port)