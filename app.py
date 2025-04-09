import streamlit as st
import pandas as pd
import pymongo
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = pymongo.MongoClient(mongo_uri)
db = client["insurance_db"]
collection = db["csv_files"]

# Load model
model = joblib.load("model.pkl")

required_cols = [
    "age", "bmi", "children", "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest"
]

# Session state flags
if "predicted_done" not in st.session_state:
    st.session_state["predicted_done"] = False
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = False

tab1, tab2 = st.tabs(["Upload CSV", "Predictions"])

with tab1:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

    if uploaded_file and not st.session_state.get("uploaded", False):
        with st.spinner("Uploading and processing file..."):
            try:
                file_name = uploaded_file.name
                df = pd.read_csv(uploaded_file)

                if not all(col in df.columns for col in required_cols):
                    st.error("Invalid CSV format. Missing required columns.")
                else:
                    existing = collection.find_one({"filename": file_name})
                    if existing:
                        collection.delete_one({"filename": file_name})

                    collection.insert_one({
                        "filename": file_name,
                        "original_csv": df.to_dict(orient="records"),
                        "predicted_csv": None
                    })

                    st.success(f"File '{file_name}' uploaded successfully.")
                    st.session_state.uploaded = True
                    st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

if "uploaded" in st.session_state and not uploaded_file:
    st.session_state.uploaded = False

with tab2:
    st.header("Predicted CSV Files Info")
    all_docs = list(collection.find())
    predicted_docs = [doc for doc in all_docs if doc.get("predicted_csv")]
    unpredicted_docs = [doc for doc in all_docs if not doc.get("predicted_csv")]

    if not all_docs:
        st.info("No files found in MongoDB.")
    else:
        if unpredicted_docs:
            if st.button("PredictAll"):
                with st.spinner("Predicting all unpredicted files..."):
                    for doc in unpredicted_docs:
                        file_name = doc["filename"]
                        original_data = pd.DataFrame(doc["original_csv"])

                        if not all(col in original_data.columns for col in required_cols):
                            st.warning(f"Skipping '{file_name}': Missing required columns.")
                            continue

                        try:
                            input_data = original_data[required_cols].values
                            predictions = model.predict(input_data)
                            original_data["prediction"] = predictions.round(2)

                            collection.update_one(
                                {"_id": doc["_id"]},
                                {"$set": {"predicted_csv": original_data.to_dict(orient="records")}}
                            )
                            st.success(f"Predictions done for '{file_name}'")
                        except Exception as e:
                            st.error(f"Error processing '{file_name}': {e}")
                st.rerun()

        for doc in predicted_docs:
            file_name = doc["filename"]
            predicted_data = pd.DataFrame(doc["predicted_csv"])

            st.markdown(f"Predictions for '{file_name}'")

            predicted_data = predicted_data.loc[:, ~predicted_data.columns.str.contains('^Unnamed|_id|^id$', case=False)]

            display_cols = required_cols + ["prediction"]
            display_cols = [col for col in display_cols if col in predicted_data.columns]

            st.dataframe(predicted_data[display_cols])
