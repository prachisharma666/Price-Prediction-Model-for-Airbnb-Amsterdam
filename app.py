import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# 1. Load your saved model and the preprocessor pipeline
# Use sm.load for statsmodels files
preprocessor = joblib.load('preprocessor.pkl')
model = sm.load('airbnb_model.pkl') 

st.set_page_config(page_title="Amsterdam Price Predictor", layout="wide")
st.title("Amsterdam Airbnb Price Predictor")

# 2. UI Inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Property Basics")
    accommodates = st.slider("Accommodates", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", 1, 10, 1)
    bathrooms = st.number_input("Bathrooms", 0.5, 10.0, 1.0)
    beds = st.number_input("Beds", 1, 20, 2)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

with col2:
    st.subheader("Location & Policy")
    # These match the neighborhoods in your OLS formula
    neighborhoods = ["Oostelijk Havengebied - Indische Buurt", "Bos en Lommer", 
                     "Buitenveldert - Zuidas", "Centrum-Oost", "De Pijp - Rivierenbuurt", 
                     "IJburg - Zeeburgereiland", "Geuzenveld - Slotermeer", 
                     "Noord-Oost", "Noord-West", "Slotervaart", "Watergraafsmeer", "Centrum-West"]
    neighbourhood = st.selectbox("Neighborhood", sorted(neighborhoods))
    min_nights = st.number_input("Minimum Nights", 1, 30, 2)
    instant_book = st.checkbox("Instant Bookable")

with col3:
    st.subheader("Reviews & History")
    clean_score = st.slider("Cleanliness Score", 0.0, 5.0, 4.8)
    loc_score = st.slider("Location Score", 0.0, 5.0, 4.5)
    val_score = st.slider("Value Score", 0.0, 5.0, 4.5)
    reviews_month = st.number_input("Reviews Per Month", 0.0, 20.0, 1.0)

if st.button("Calculate Estimated Price"):
    # 3. Build input dict matching raw column names from your original CSV
    input_dict = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms_count": bathrooms,
        "beds": beds,
        "minimum_nights": min_nights,
        "review_scores_cleanliness": clean_score,
        "review_scores_location": loc_score,
        "review_scores_value": val_score,
        "instant_bookable": 1 if instant_book else 0,
        "host_identity_verified": 1, 
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood,
        "availability_365": 100,
        "reviews_per_month": reviews_month,
        "first_review_days": 100
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # 4. Transform using Pipeline (Scaling and One-Hot Encoding)
    processed_data = preprocessor.transform(input_df)
    
    # Get feature names from preprocessor and clean them
    # e.g., 'cat__room_type_Private room' becomes 'room_type_Private room'
    raw_cols = [c.split("__")[-1] for c in preprocessor.get_feature_names_out()]
    processed_df = pd.DataFrame(processed_data, columns=raw_cols)

    # 5. MANUALLY CONSTRUCT FORMULA TERMS (Crucial for Statsmodels consistency)
    
    # Squared Terms
    processed_df['accommodates2'] = processed_df['accommodates'] ** 2
    processed_df['minimum_nights2'] = processed_df['minimum_nights'] ** 2
    
    # Categorical Interaction: accommodates * Q('room_type_Private room')
    # We find the 0/1 flag for Private room and multiply it by accommodates
    is_private = processed_df['room_type_Private room'] if 'room_type_Private room' in processed_df.columns else 0
    processed_df["accommodates:Q('room_type_Private room')"] = processed_df['accommodates'] * is_private
    
    # Numerical Interaction: accommodates * bedrooms
    processed_df['accommodates:bedrooms'] = processed_df['accommodates'] * processed_df['bedrooms']

    # 6. Format Neighborhood columns to match the OLS "Q('name')" syntax
    # Statsmodels formula API renames categorical columns with Q() for safety
    for col in processed_df.columns:
        if "neighbourhood_cleansed" in col:
            processed_df.rename(columns={col: f"Q('{col}')"}, inplace=True)

    # 7. Match exactly with Model Parameter Index
    model_columns = model.params.index.tolist()
    
    # Add Intercept if required by model
    if 'Intercept' in model_columns:
        processed_df['Intercept'] = 1.0
        
    # Reindex ensures all expected columns exist (fills missing neighborhoods with 0)
    # and puts them in the exact order the model expects.
    final_input = processed_df.reindex(columns=model_columns, fill_value=0.0)

    # 8. Predict and Invert Log
    log_pred = model.predict(final_input)
    # Use expm1 to reverse the log1p transformation used during training
    final_price = np.expm1(log_pred[0])

    # Display Result
    st.divider()
    st.success(f"### Estimated Price: ${final_price:.2f} per night")
    st.info("Note: This prediction accounts for interactions between property size and room type.")
