import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# Load the trained model and encoders
# -------------------------------
model = pickle.load(open("crop_yield_model.pkl", "rb"))
le_crop = pickle.load(open("le_crop.pkl", "rb"))
le_state = pickle.load(open("le_state.pkl", "rb"))
le_season = pickle.load(open("le_season.pkl", "rb"))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Prediction Dashboard")
st.markdown("### Predict Crop Yield, Revenue, or Price based on inputs")

# -------------------------------
# Define main() function
# -------------------------------
def main():
    crop_name = st.text_input("Enter Crop Name (e.g. Rice, Wheat, Maize)")
    state_name = st.text_input("Enter State Name (e.g. Karnataka, Punjab, Maharashtra)")
    season_name = st.text_input("Enter Season Name (e.g. Kharif, Rabi, Whole Year)")

    area = st.number_input("Area (in Hectares)", min_value=0.0, step=0.1)
    rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0.0, step=0.1)
    fertilizer = st.number_input("Fertilizer Used (in kg/ha)", min_value=0.0, step=0.1)
    pesticide = st.number_input("Pesticide Used (in kg/ha)", min_value=0.0, step=0.1)

    if st.button("🔍 Predict Yield"):
        try:
            crop_encoded = le_crop.transform([crop_name])[0]
            state_encoded = le_state.transform([state_name])[0]
            season_encoded = le_season.transform([season_name])[0]

            test_data = pd.DataFrame({
                "Crop": [crop_encoded],
                "State": [state_encoded],
                "Season": [season_encoded],
                "Area": [area],
                "Annual_Rainfall": [rainfall],
                "Fertilizer": [fertilizer],
                "Pesticide": [pesticide]
            })

            pred = model.predict(test_data)
            st.success(f"🌾 Predicted Yield: {pred[0]:.2f}")

        except Exception as e:
            st.error("⚠️ The crop/state/season you entered was not found in training data.")
            st.info("Try using names exactly as in the dataset (e.g. Rice, Wheat, etc.)")
            st.text(e)

    st.markdown("---")
    st.caption("👩‍🌾 Built by Tanmay | Powered by Streamlit + ML")

# -------------------------------
# Run main()
# -------------------------------
if __name__ == "__main__":
    main()
