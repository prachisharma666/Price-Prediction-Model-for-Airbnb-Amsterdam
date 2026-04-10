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

    min_nights = st.number_input("Minimum Nights", 1, 30, 1)

    review_score = st.slider("Cleanliness Score", 0.0, 5.0, 4.5)

    instant_book = st.checkbox("Instant Bookable")

    verified = st.checkbox("Host Identity Verified")



# 2. Process Input

if st.button("Predict Price"):

    # Build a dictionary matching your training column names

    input_dict = {

        "accommodates": accommodates,

        "bedrooms": bedrooms,

        "bathrooms_count": bathrooms,

        "minimum_nights": min_nights,

        "review_scores_cleanliness": review_score,

        "instant_bookable": 1 if instant_book else 0,

        "host_identity_verified": 1 if verified else 0,

        "room_type": room_type,

        # Default values for columns not in UI but required by preprocessor

        "latitude": 52.3, "longitude": 4.9, "beds": 1, "maximum_nights": 30,

        "availability_365": 100, "number_of_reviews": 10, "reviews_per_month": 1.0,

        "review_scores_value": 4.5, "review_scores_location": 4.5, "review_scores_rating": 4.5,

        "neighbourhood_cleansed": "Centrum-West", "host_is_superhost": 0, "has_availability": 1,

        "first_review_days": 100, "last_review_days": 10

    }

    

    input_df = pd.DataFrame([input_dict])

    

    # Apply Preprocessing

    processed_df = preprocessor.transform(input_df)

    processed_df.columns = [c.split("__")[-1] for c in processed_df.columns]

    

    # Feature Engineering (Squares)

    processed_df['accommodates2'] = processed_df['accommodates'] ** 2

    processed_df['minimum_nights2'] = processed_df['minimum_nights'] ** 2



    # Prediction

    log_pred = model.predict(processed_df)

    final_price = np.expm1(log_pred[0])



    st.success(f"Estimated Nightly Price: ${final_price:.2f}")
