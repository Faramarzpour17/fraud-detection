import streamlit as st
import pandas as pd
import joblib

# 1. Load your actual trained AI engine
@st.cache_resource
def load_model():
    return joblib.load('explainable_fraud_model_tuned.pkl')

pipeline = load_model()

# 2. Build the User Interface
st.title("🚨 Fraud Detection AI Simulator")
st.markdown("Adjust the transaction parameters below to see how the AI evaluates risk in real-time.")

# 3. Create Sliders for your 8 engineered features
amt = st.slider("Transaction Amount ($)", 1.0, 5000.0, 50.0)
distance_km = st.slider("Distance from Home (km)", 0.0, 2000.0, 5.0)
trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
customer_age = st.slider("Customer Age", 18, 100, 35)

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Merchant Category", ['grocery_pos', 'shopping_net', 'gas_transport', 'misc_net'])
    gender = st.selectbox("Gender", ['M', 'F'])
with col2:
    city_pop = st.number_input("City Population", value=100000)
    trans_day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

# 4. Process the data and predict
if st.button("Evaluate Transaction"):
    # Pack the user input into a dataframe
    input_data = pd.DataFrame([[category, amt, gender, city_pop, distance_km, trans_hour, trans_day_of_week, customer_age]], 
                              columns=['category', 'amt', 'gender', 'city_pop', 'distance_km', 'trans_hour', 'trans_day_of_week', 'customer_age'])
    
    # Get the fraud probability
    fraud_prob = pipeline.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    
    # THE HYBRID DEFENSE: Hard-coded Business Rules catch what the AI misses
    if category == 'gas_transport' and amt > 500:
        st.error("STATUS: BLOCKED 🛑 | Alert: Rule-Based Heuristic Override (Impossible Category Amount)")
    elif fraud_prob > 50:
        st.error(f"STATUS: BLOCKED 🛑 | AI Fraud Probability: {fraud_prob:.2f}%")
    else:
        st.success(f"STATUS: APPROVED ✅ | AI Fraud Probability: {fraud_prob:.2f}%")
    
    # Get the fraud probability
    fraud_prob = pipeline.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    if fraud_prob > 50:
        st.error(f"STATUS: BLOCKED 🛑 | Fraud Probability: {fraud_prob:.2f}%")
    else:
        st.success(f"STATUS: APPROVED ✅ | Fraud Probability: {fraud_prob:.2f}%")
