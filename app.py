import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")  # Ensure the model file is in the same directory

# Streamlit App
st.title("Predict Insurance Charges")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ["age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]
    
    if all(col in df.columns for col in required_cols):
        # Make predictions
        predictions = model.predict(df[required_cols].values)
        
        # Add predictions to DataFrame
        df["prediction"] = predictions.round(2)

        # Display predictions
        st.subheader("Predictions:")
        st.write(df[["prediction"]])

        # Option to download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    
    else:
        st.error("Invalid CSV format. Missing required columns.")
