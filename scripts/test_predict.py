import joblib
import pandas as pd
import numpy as np
import os
import sys

def main():
    model_path = os.path.join("models", "model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    features_path = os.path.join("models", "features.pkl")

    for p in (model_path, scaler_path, features_path):
        print(f"Checking: {p} ->", os.path.exists(p))

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
    except Exception as e:
        print("Error loading artifacts:", e)
        sys.exit(1)

    # Ensure features is a list
    if not isinstance(features, list):
        try:
            features = list(features)
        except Exception:
            features = [str(f) for f in features]

    # Build sample input matching the app
    input_dict = {
        "mileage_kmpl": 18.0,
        "engine_cc": 1600.0,
        "owner_count": 0.0,
        "accidents_reported": 0.0,
        "car_age": 5.0
    }

    df_input = pd.DataFrame([input_dict])
    # set a brand example
    brand_col = [c for c in features if c.startswith('brand_')]
    if brand_col:
        df_input[brand_col[0]] = 1
    # set fuel/transmission/service if present
    fuel_col = [c for c in features if c.startswith('fuel_type_')]
    if fuel_col:
        df_input[fuel_col[0]] = 0
    trans_col = [c for c in features if c.startswith('transmission_')]
    if trans_col:
        df_input[trans_col[0]] = 0
    serv_col = [c for c in features if c.startswith('service_history_')]
    if serv_col:
        df_input[serv_col[-1]] = 1

    df_final = df_input.reindex(columns=features, fill_value=0)
    df_final = df_final.astype(float)

    print("\nAligned input (unscaled):")
    print(df_final.T.to_string())

    if hasattr(scaler, 'mean_'):
        print('\nscaler.mean_ (first 20):', list(np.round(scaler.mean_[:20], 6)))
    if hasattr(scaler, 'scale_'):
        print('scaler.scale_ (first 20):', list(np.round(scaler.scale_[:20], 6)))

    try:
        X_scaled = scaler.transform(df_final)
        pred = model.predict(X_scaled)[0]
        print(f"\nPredicted Price: ${pred:,.2f}")
    except Exception as e:
        print("Error during scaling/prediction:", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
