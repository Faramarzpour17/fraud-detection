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
    # Your 4 primary app categories
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
    
    # Get the AI's predicted fraud probability
    fraud_prob = pipeline.predict_proba(input_data)[0][1] * 100
    
    st.markdown("---")
    
    # ==========================================
    # LAYER 2: HARD-CODED HYBRID DEFENSE RULES
    # ==========================================
    
    # Rule 1: Grocery Guardrail
    if category == 'grocery_pos' and amt > 1500:
        st.error("STATUS: BLOCKED 🛑 | Rule Override: Impossible Grocery Amount (> $1,500)")
        
    # Rule 2: Gas & Transport Guardrail
    elif category == 'gas_transport' and amt > 500:
        st.error("STATUS: BLOCKED 🛑 | Rule Override: Impossible Gas/Transport Amount (> $500)")
        
    # Rule 3: Online Shopping Guardrail
    elif category == 'shopping_net' and amt > 3500:
        st.error("STATUS: BLOCKED 🛑 | Rule Override: Excessive Online Shopping Target (> $3,500)")
        
    # Rule 4: Misc Online Guardrail (High Risk)
    elif category == 'misc_net' and amt > 2000:
        st.error("STATUS: BLOCKED 🛑 | Rule Override: Suspicious Misc. Web Transfer (> $2,000)")
        
    # Rule 5: The "Vampire Hour" Rule (Cross-category)
    elif amt > 1000 and (1 <= trans_hour <= 5) and distance_km > 200:
         st.error("STATUS: BLOCKED 🛑 | Rule Override: High-Value, Late-Night, Long-Distance Anomaly")
         
    # ==========================================
    # LAYER 1: RANDOM FOREST AI (If rules pass)
    # ==========================================
    elif fraud_prob > 50:
        st.error(f"STATUS: BLOCKED 🛑 | AI Fraud Probability: {fraud_prob:.2f}%")
    else:
        st.success(f"STATUS: APPROVED ✅ | AI Fraud Probability: {fraud_prob:.2f}%")
