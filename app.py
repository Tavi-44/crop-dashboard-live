import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os

# --- 1. Data Loading and Model Training ---
@st.cache_data
def load_data():
    file_path = "crop_revenue.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error(f"Error: File '{file_path}' not found. Please ensure the CSV file is in the same directory as this script.")
        st.stop()
    
    # Data Cleaning and Transformation for the model
    df_model = df.copy()
    # Only keep positive values for log transformation
    df_model = df_model[(df_model['Area'] > 0) & (df_model['Production'] > 0) & (df_model['Fertilizer'] > 0)]
    
    df_model['log_Area'] = np.log(df_model['Area'])
    df_model['log_Production'] = np.log(df_model['Production'])
    df_model['log_Fertilizer'] = np.log(df_model['Fertilizer']) # Log transform Fertilizer for better feature handling
    
    return df, df_model # Return both original and transformed data

@st.cache_resource
def train_model(df_model):
    # Features (X) aur Target (y) define karna. Log Fertilizer is now a feature.
    features = ['log_Area', 'Annual_Rainfall', 'log_Fertilizer', 'Pesticide'] 
    target = 'log_Production'

    X = df_model[features]
    y = df_model[target]

    # Random Forest Regressor Model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X, y)

    # Performance check
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    st.sidebar.caption(f"**Model Type:** Advanced & Smart Prediction (Random Forest)")
    # R2 Score ko simplify karna
    r2_status = "Very High" if r2 > 0.8 else "Good" if r2 > 0.5 else "Low"
    st.sidebar.caption(f"**Prediction Reliability:** {r2_status} ({r2:.2f})")

    return model, feature_importances

# --- 2. Streamlit UI Design ---
def run_app():
    st.set_page_config(
        page_title="Dynamic Crop Production Dashboard (Client View) 📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Advanced Crop Production & Revenue Dashboard")
    st.markdown("🚀 Based on your inputs, this dashboard provides a **highly reliable prediction** of **Production** and **Revenue**.")

    # Data load aur Model train karna
    data_original, data_model = load_data()
    model, feature_importances = train_model(data_model)

    st.sidebar.header("Input Parameters (Inhe Change Karein)")

    # --- Sidebar Inputs ---
    area = st.sidebar.slider("Area (Hectares)",
                             min_value=float(data_original['Area'].min()),
                             max_value=float(data_original['Area'].max()),
                             value=float(data_original['Area'].median()),
                             step=1000.0)

    annual_rainfall = st.sidebar.slider("Annual Rainfall (mm)",
                                        min_value=float(data_original['Annual_Rainfall'].min()),
                                        max_value=float(data_original['Annual_Rainfall'].max()),
                                        value=float(data_original['Annual_Rainfall'].median()),
                                        step=100.0)

    fertilizer = st.sidebar.slider("Fertilizer (kg)",
                                   min_value=float(data_original['Fertilizer'].min()),
                                   max_value=float(data_original['Fertilizer'].max()),
                                   value=float(data_original['Fertilizer'].median()),
                                   step=10000.0)

    pesticide = st.sidebar.slider("Pesticide (kg)",
                                  min_value=float(data_original['Pesticide'].min()),
                                  max_value=float(data_original['Pesticide'].max()),
                                  value=float(data_original['Pesticide'].median()),
                                  step=1000.0)

    # --- 3. Prediction ---
    # Log transform inputs
    log_area = np.log(area) if area > 0 else np.log(1)
    log_fertilizer = np.log(fertilizer) if fertilizer > 0 else np.log(1)
    
    input_data = pd.DataFrame([[log_area, annual_rainfall, log_fertilizer, pesticide]],
                              columns=['log_Area', 'Annual_Rainfall', 'log_Fertilizer', 'Pesticide'])

    log_prediction = model.predict(input_data)[0]
    prediction = np.exp(log_prediction)

    st.header("Predicted Production & Revenue Estimate")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Predicted Crop Production",
                  value=f"{prediction:,.0f} Units")

    with col2:
        arbitrary_price = 100
        estimated_revenue = prediction * arbitrary_price
        st.metric(label=f"Estimated Revenue (at assumed Price of ₹ {arbitrary_price}/Unit)",
                  value=f"₹ {estimated_revenue:,.0f}")

    st.markdown("---")

    # --- 4. Dynamic Charts ---
    st.header("What-If Analysis & Key Factors (Live Update)")

    col3, col4 = st.columns(2)
    
    # --- Col 3: Fertilizer Trend (Line Plot) ---
    with col3:
        st.subheader("Impact of Increasing Fertilizer on Production (What-If)")

        # Range for Fertilizer to plot (100 points)
        min_fert = data_original['Fertilizer'].min()
        max_fert = data_original['Fertilizer'].max()
        fertilizer_range = np.linspace(min_fert, max_fert, 100)
        
        # Log transform the range
        log_fertilizer_range = np.log(fertilizer_range)
        
        # Prediction DataFrame banana (baaki inputs constant rakhe gaye hain)
        X_trend = pd.DataFrame({
            'log_Area': [log_area] * 100,  
            'Annual_Rainfall': [annual_rainfall] * 100,
            'log_Fertilizer': log_fertilizer_range, 
            'Pesticide': [pesticide] * 100   
        })
        
        # Predictions
        log_preds = model.predict(X_trend)
        preds = np.exp(log_preds)
        
        trend_df = pd.DataFrame({'Fertilizer (kg)': fertilizer_range, 'Predicted Production (Units)': preds})
        
        fig_line = px.line(
            trend_df,
            x='Fertilizer (kg)',
            y='Predicted Production (Units)',
            title='Production Trend as Fertilizer Changes',
            labels={'Fertilizer (kg)': 'Fertilizer (kg)', 'Predicted Production (Units)': 'Predicted Production (Units)'}
        )
        # Current Fertilizer value ko highlight karna
        fig_line.add_vline(x=fertilizer, line_width=2, line_dash="dash", line_color="red", name="Current Input")

        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("**Live Analysis:**The red dashed line indicates your current Fertilizer input. Observe how changing the Fertilizer affects the Production..")

    # --- Col 4: Feature Importance (Static) ---
    with col4:
        st.subheader("Which factor is more important? 🧠")
        # Feature Importance Plot (Static but Crucial for client)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        
        # Clean labels for the client
        importance_df = feature_importances.rename({
            'log_Area': 'Area (Log)', 
            'log_Fertilizer': 'Fertilizer (Log)'
        })
        
        ax_imp.barh(importance_df.index, importance_df.values, color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        ax_imp.set_title('Contribution of Each Factor to Prediction', fontsize=14)
        ax_imp.set_xlabel('Relative Importance Score', fontsize=12)
        plt.gca().invert_yaxis() # Top factor upar
        plt.tight_layout()
        st.pyplot(fig_imp)
        st.caption("**Analysis:**The longer the bar for a factor, the more **important** it is for the **Production Prediction**.")


# Run the Streamlit application
if __name__ == "__main__":
    run_app()