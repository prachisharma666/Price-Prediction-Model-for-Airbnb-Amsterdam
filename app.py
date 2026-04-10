import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# Load model and preprocessor
preprocessor = joblib.load('preprocessor.pkl')
model = sm.load('airbnb_model.pkl')

st.title("Amsterdam Airbnb Price Predictor")
st.write("Enter listing details to estimate the nightly price.")

# 1. Create Input Fields
col1, col2 = st.columns(2)

with col1:
    accommodates = st.slider("Accommodates", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", 0, 10, 1)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

with col2:
    neighborhood_list = [
        "Centrum-West", "Centrum-Oost", "De Pijp - Rivierenbuurt", 
        "Zuid", "Oud-West", "Bos en Lommer", "Westerpark", 
        "Oost", "Watergraafsmeer", "Oud-Oost", "Noord-West", 
        "Noord-Oost", "Slotervaart", "Osdorp", "Buitenveldert - Zuidas",
        "Oostelijk Havengebied - Indische Buurt", "IJburg - Zeeburgereiland",
        "Geuzenveld - Slotermeer", "Bijlmer-Centrum", "Bijlmer-Oost", "Gaasperdam - Driemond"
    ]
    neighbourhood = st.selectbox("Neighbourhood", sorted(neighborhood_list))
    min_nights = st.number_input("Minimum Nights", 1, 30, 1)
    clean_score = st.slider("Cleanliness Score", 0.0, 5.0, 4.5)
    instant_book = st.checkbox("Instant Bookable")

# 2. Process Input
if st.button("Predict Price"):
    input_dict = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms_count": bathrooms,
        "minimum_nights": min_nights,
        "review_scores_cleanliness": clean_score,
        "instant_bookable": 1 if instant_book else 0,
        "host_identity_verified": 1,
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood,
        "latitude": 52.3, "longitude": 4.9, "beds": 1, "maximum_nights": 30,
        "availability_365": 100, "number_of_reviews": 10, "reviews_per_month": 1.0,
        "review_scores_value": 4.5, "review_scores_location": 4.5, "review_scores_rating": 4.5,
        "host_is_superhost": 0, "has_availability": 1,
        "first_review_days": 100, "last_review_days": 10
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Apply Preprocessing
    processed_data = preprocessor.transform(input_df)
    cols = [c.split("__")[-1] for c in preprocessor.get_feature_names_out()]
    processed_df = pd.DataFrame(processed_data, columns=cols)
    
    # --- FEATURE ENGINEERING ---
    processed_df['accommodates2'] = processed_df['accommodates'] ** 2
    processed_df['minimum_nights2'] = processed_df['minimum_nights'] ** 2
    processed_df['accommodates:bedrooms'] = processed_df['accommodates'] * processed_df['bedrooms']
    
    is_private = 1.0 if room_type == "Private room" else 0.0
    processed_df["accommodates:Q('room_type_Private room')"] = processed_df['accommodates'] * is_private

    # Rename Neighborhoods to the Q('') format your model expects
    for col in processed_df.columns:
        if "neighbourhood_cleansed" in col:
            processed_df.rename(columns={col: f"Q('{col}')"}, inplace=True)

    # --- THE FIX: MANUAL PREDICTION ---
    # 1. Get the list of features (weights) the model expects
    model_params = model.params.index.tolist()
    if 'Intercept' in model_params:
        processed_df['Intercept'] = 1.0
    
    # 2. Reindex to ensure order matches model.params exactly
    final_input = processed_df.reindex(columns=model_params, fill_value=0.0)

    # 3. Multiply weights by inputs (Manual Dot Product)
    # This avoids the Patsy formula check entirely
    log_pred = np.dot(final_input.iloc[0].values, model.params.values)
    
    # 4. Final conversion
    final_price = np.expm1(log_pred)

    st.success(f"Estimated Nightly Price: ${final_price:.2f}")
