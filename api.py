from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="assets", template_folder=".")
CORS(app)

# --- KONFIGURASI PATH MODEL ---
# Pastikan file .pkl ada di folder yang sama atau sesuaikan path-nya
# --- KONFIGURASI PATH MODEL ---
MODEL_PATH = 'rf_brfss_diabetes_binary.pkl'


# --- LOAD MODEL (Hanya sekali saat server start) ---
try:
    if os.path.exists(MODEL_PATH):
        # Load dictionary yang disimpan dari training
        model_data = joblib.load(MODEL_PATH)
        
        # Ambil model asli dari dalam dictionary
        # Fallback: Jika ternyata user menyimpan raw model (bukan dict), handle errornya
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            feature_names = model_data.get("feature_names", [])
            print(f"✅ Model loaded successfully.")
            print(f"📋 Model expects {len(feature_names)} features: {feature_names}")
        else:
            model = model_data
            print("⚠️ Warning: Model loaded as raw object (No feature names metadata).")
    else:
        model = None
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Fatal Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Cek Ketersediaan Model
    if not model:
        return jsonify({'error': 'Model not loaded properly on server'}), 500

    try:
        # 2. Ambil Data JSON dari Frontend
        data = request.json
        print("📥 Received Data:", data) # Debug log

        # 3. Validasi & Kalkulasi BMI
        # Frontend mengirim 'weight' (kg) dan 'height' (cm)
        try:
            weight = float(data.get('weight', 0))
            height_cm = float(data.get('height', 0))
            
            if height_cm > 0:
                height_m = height_cm / 100
                bmi = weight / (height_m ** 2)
            else:
                bmi = 0 # Fallback jika tinggi 0 (hindari division by zero)
        except ValueError:
            return jsonify({'error': 'Invalid Weight/Height format'}), 400

        # 4. MAPPING INPUT FRONTEND -> FITUR MODEL
        # Urutan ini WAJIB SAMA PERSIS dengan X.columns di Training
        # Berdasarkan untitled20 (4).py, urutan kolom setelah drop adalah:
        # HighBP, HighChol, BMI, Smoker, PhysActivity, GenHlth, DiffWalk, 
        # Sex, Age, Education, Income, NoDocbcCost, HeartHistory
        
        features = [
            float(data.get('HighBP', 0)),               # 1. HighBP
            float(data.get('HighChol', 0)),             # 2. HighChol
            float(bmi),                                 # 3. BMI (Hasil Hitungan)
            float(data.get('Smoker', 0)),               # 4. Smoker
            float(data.get('PhysActivity', 0)),         # 5. PhysActivity
            float(data.get('GenHlth', 3)),              # 6. GenHlth (Default 3/Netral)
            float(data.get('DiffWalk', 0)),             # 7. DiffWalk
            float(data.get('Sex', 0)),                  # 8. Sex
            float(data.get('age', 1)),                  # 9. Age (Perhatikan huruf kecil 'age' dari frontend)
            float(data.get('education', 4)),            # 10. Education
            float(data.get('income', 5)),               # 11. Income
            float(data.get('NoDoc', 0)),                # 12. NoDocbcCost (Mapping dari 'NoDoc')
            float(data.get('HeartAttackOrStroke', 0))   # 13. HeartHistory (Mapping dari 'HeartAttackOrStroke')
        ]

        # 5. Konversi ke Numpy Array (Bentuk 2D)
        final_features = np.array([features])
        
        # Debugging: Cek apa yang masuk ke model
        # print(f"Features sent to model: {final_features}")

        # 6. Prediksi
        # output predict_proba bentuknya [[Prob_0, Prob_1]]
        probability = model.predict_proba(final_features)
        
        # Ambil probabilitas kelas 1 (Risiko Diabetes)
        risk_score = probability[0][1] 

        # 7. Return Response
        return jsonify({
            'prediction': int(risk_score > 0.5), # 0 atau 1 (Threshold 50%)
            'risk_score': float(risk_score)      # Nilai desimal (cth: 0.75)
        })

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error lengkap ke terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)