import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('airbnb_model.pkl') 

st.set_page_config(page_title="Amsterdam Price Predictor", layout="wide")
st.title("Amsterdam Airbnb Price Predictor")
st.write("Adjust the features below to see how they impact the nightly price.")

neighborhoods = [
    "Centrum-West", "Centrum-Oost", "De Pijp - Rivierenbuurt", 
    "Zuid", "Oud-West", "De Baarsjes - Oud-West", "Bos en Lommer",
    "Westerpark", "Oost", "Watergraafsmeer", "Oud-Oost",
    "Indische Buurt - Oostelijk Havengebied", "Noord-West", 
    "Noord-Oost", "Slotervaart", "Osdorp", "Buitenveldert - Zuidas",
    "Gaasperdam - Driemond", "Bijlmer-Centrum", "Bijlmer-Oost",
    "IJburg - Zeeburgereiland", "Noord-West"
]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Property Basics")
    accommodates = st.slider("Accommodates", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", 1, 10, 1)
    bathrooms = st.number_input("Bathrooms", 0.5, 10.0, 1.0, step=0.5)
    beds = st.number_input("Beds", 1, 20, 1)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

with col2:
    st.subheader("Location & Policy")
    neighbourhood = st.selectbox("Neighborhood", sorted(neighborhoods))
    min_nights = st.number_input("Minimum Nights", 1, 30, 2)
    instant_book = st.checkbox("Instant Bookable")
    verified = st.checkbox("Host Identity Verified")
    superhost = st.checkbox("Host is Superhost")

with col3:
    st.subheader("Reviews & Availability")
    clean_score = st.slider("Cleanliness Score", 0.0, 5.0, 4.8)
    rating_score = st.slider("Overall Rating", 0.0, 5.0, 4.7)
    availability = st.slider("Availability (Days/Year)", 0, 365, 100)
    reviews_month = st.number_input("Reviews Per Month", 0.0, 20.0, 1.0)

if st.button("Calculate Estimated Price"):
    input_dict = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms_count": bathrooms,
        "beds": beds,
        "minimum_nights": min_nights,
        "review_scores_cleanliness": clean_score,
        "review_scores_rating": rating_score,
        "instant_bookable": "t" if instant_book else "f",
        "host_identity_verified": "t" if verified else "f",
        "host_is_superhost": "t" if superhost else "f",
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood,
        "availability_365": availability,
        "reviews_per_month": reviews_month,
        # Filling defaults for secondary features not in UI to prevent errors
        "latitude": 52.37, "longitude": 4.89, "maximum_nights": 30,
        "number_of_reviews": 10, "review_scores_value": 4.5, 
        "review_scores_location": 4.5, "has_availability": "t",
        "first_review_days": 100, "last_review_days": 10
    }
    
    input_df = pd.DataFrame([input_dict])
    
    processed_data = preprocessor.transform(input_df)
    
    processed_df = pd.DataFrame(processed_data, columns=[c.split("__")[-1] for c in preprocessor.get_feature_names_out()])
    
    processed_df['accommodates2'] = processed_df['accommodates'] ** 2
    processed_df['bedrooms2'] = processed_df['bedrooms'] ** 2
    processed_df['minimum_nights2'] = processed_df['minimum_nights'] ** 2

    cols_when_model_builds = model.params.index.tolist()
    if 'Intercept' in cols_when_model_builds:
        processed_df['Intercept'] = 1.0
        processed_df = processed_df[cols_when_model_builds]
    else:
        processed_df = processed_df[cols_when_model_builds]

    log_pred = model.predict(processed_df)

    final_price = np.expm1(log_pred[0])

    st.divider()
    st.metric(label="Predicted Nightly Price", value=f"${final_price:.2f}")
    
    if final_price > 300:
        st.warning("This listing is priced significantly above the market average.")
    else:
        st.success("This prediction is based on current Amsterdam market trends.")
