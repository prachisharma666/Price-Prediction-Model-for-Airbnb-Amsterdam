import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# 1. Load your tools
preprocessor = joblib.load('preprocessor.pkl')
model = sm.load('airbnb_model.pkl') 

st.title("Amsterdam Airbnb Price Predictor")

# 2. UI Inputs (Simplified for brevity)
accommodates = st.slider("Accommodates", 1, 16, 2)
bedrooms = st.number_input("Bedrooms", 1, 10, 1)
bathrooms = st.number_input("Bathrooms", 0.5, 10.0, 1.0)
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
neighbourhood = st.selectbox("Neighborhood", ["Centrum-Oost", "Bos en Lommer", "Noord-West", "Other"])

if st.button("Predict"):
    # 3. Create raw input
    input_dict = {
        "accommodates": accommodates, "bedrooms": bedrooms, "bathrooms_count": bathrooms,
        "room_type": room_type, "neighbourhood_cleansed": neighbourhood,
        "beds": 1, "minimum_nights": 2, "review_scores_cleanliness": 4.5,
        "review_scores_location": 4.5, "review_scores_value": 4.5,
        "instant_bookable": 1, "host_identity_verified": 1, "availability_365": 100,
        "reviews_per_month": 1.0, "latitude": 52.3, "longitude": 4.9,
        "maximum_nights": 30, "number_of_reviews": 10, "review_scores_rating": 4.5,
        "host_is_superhost": 0, "has_availability": 1, "first_review_days": 100, "last_review_days": 10
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # 4. Transform using Pipeline
    processed_data = preprocessor.transform(input_df)
    raw_cols = [c.split("__")[-1] for c in preprocessor.get_feature_names_out()]
    processed_df = pd.DataFrame(processed_data, columns=raw_cols)

    # 5. Create the columns exactly as they appear in your OLS summary
    processed_df['accommodates2'] = processed_df['accommodates'] ** 2
    processed_df['minimum_nights2'] = processed_df['minimum_nights'] ** 2
    processed_df['accommodates:bedrooms'] = processed_df['accommodates'] * processed_df['bedrooms']
    
    # Interaction term name must match the OLS summary exactly
    is_private = 1.0 if room_type == "Private room" else 0.0
    processed_df["accommodates:Q('room_type_Private room')"] = processed_df['accommodates'] * is_private

    # Rename neighborhoods to match the Q('') format in the model
    for col in processed_df.columns:
        if "neighbourhood_cleansed" in col:
            processed_df.rename(columns={col: f"Q('{col}')"}, inplace=True)

    # 6. Final Alignment
    model_params = model.params.index.tolist()
    if 'Intercept' in model_params:
        processed_df['Intercept'] = 1.0
    
    # Reindex fills missing values with 0 and matches the order
    final_input = processed_df.reindex(columns=model_params, fill_value=0.0)

    # 7. THE FIX: Tell statsmodels this is a simple array prediction
    # By using the .values, we keep the order but stop the "Patsy" logic from triggering
# Pass the data as a raw array (.values) 
# This tells Statsmodels: "Don't check the names, just use the numbers in this order."
    log_pred = model.predict(exog=final_input.values)
    
    # Then finish as usual
    final_price = np.expm1(log_pred[0])
    st.success(f"Price: ${final_price:.2f}")
