import streamlit as st
import pandas as pd
import os
import joblib

# Load model
model = joblib.load("model.pkl")

required_cols = [
    "age", "bmi", "children", "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest"
]

CSV_FOLDER = "csv_files"
PREDICTED_FOLDER = "predicted_files"

os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

st.title("Insurance Prediction Dashboard")

csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]
predicted_files = [f for f in os.listdir(PREDICTED_FOLDER) if f.endswith(".csv")]

unpredicted_files = [f for f in csv_files if f not in predicted_files]

if unpredicted_files:
    if st.button("PredictAll"):
        for file in unpredicted_files:
            file_path = os.path.join(CSV_FOLDER, file)
            predicted_path = os.path.join(PREDICTED_FOLDER, file)

            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in required_cols):
                    st.warning(f"Skipping '{file}': Missing required columns.")
                    continue

                input_data = df[required_cols]
                predictions = model.predict(input_data)
                df["prediction"] = predictions.round(2)

                df.to_csv(predicted_path, index=False)
                st.success(f"Predictions saved for '{file}'")
            except Exception as e:
                st.error(f"Error processing '{file}': {e}")
        st.rerun() 

st.subheader("Predicted CSV Files")
if not predicted_files:
    st.info("No predicted files found.")
else:
    for file in predicted_files:
        file_path = os.path.join(PREDICTED_FOLDER, file)
        try:
            df = pd.read_csv(file_path)
            st.markdown(f"Predictions for `{file}`")
            display_cols = [col for col in required_cols + ["prediction"] if col in df.columns]
            st.dataframe(df[display_cols])
        except Exception as e:
            st.error(f"Could not display predictions for '{file}': {e}")
