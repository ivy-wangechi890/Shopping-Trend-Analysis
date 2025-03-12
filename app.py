import streamlit as st
import joblib
import numpy as np

# Load Models
@st.cache_resource
def load_clv_model():
    return joblib.load('lr_cv.pkl')

@st.cache_resource
def load_churn_model():
    return joblib.load('gb.pkl')

st.cache_resource.clear()

clv_model = load_clv_model()
churn_model = load_churn_model()

# Custom Styling - White Background & Black-Themed Elements
st.markdown("""
    <style>
        body { background-color: #FFFFFF; color: #000000; font-family: 'Arial', sans-serif; }
        .stApp { background-color: #FFFFFF; color: #000000; }
        .stButton>button { background-color: #000000; color: white; font-size: 16px; padding: 12px; border-radius: 8px; border: none; }
        .stButton>button:hover { background-color: #444; }
        .stTextInput>div>input, .stNumberInput input, .stSelectbox select { 
            border: 1px solid #000000; border-radius: 5px; padding: 8px; font-size: 14px; background-color: #F5F5F5; color: black;
        }
        h1, h2, h3, h4 { color: #000000; }
        .result-box { background-color: #000000; color: white; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Customer Insights Tool")
st.markdown("""
### Welcome to the Customer Analytics Platform
This website is designed to predict **Customer Lifetime Value (CLV)** and **Customer Churn** using machine learning.  
By analyzing customer behavior, purchasing patterns, and engagement metrics, it provides **actionable insights** to help businesses:
- Improve **customer retention**
- Optimize **marketing strategies**
- Maximize **revenue growth**
""")

st.markdown("---")

# Input Form
st.header("Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    Gender_encoded = st.selectbox("Gender", [(0, "Male"), (1, "Female")], format_func=lambda x: x[1])[0]
    ItemPurchased_encoded = st.selectbox("Item Purchased", [
        (1, "Blouse"), (2, "Jewelry"), (3, "Pants"), (4, "Shirt"), (5, "Dress"),
        (6, "Sweater"), (7, "Jacket"), (8, "Belt"), (9, "Sunglasses"), (10, "Coat"),
        (11, "Sandals"), (12, "Socks"), (13, "Skirt"), (14, "Shorts"), (15, "Scarf"),
        (16, "Hat"), (17, "Handbag"), (18, "Hoodie"), (19, "Shoes"), (20, "T-shirt"),
        (21, "Sneakers"), (22, "Boots"), (23, "Backpack"), (24, "Gloves"), (25, "Jeans")
    ], format_func=lambda x: x[1])[0]

    Category_encoded = st.selectbox("Category", [(1, "Clothing"), (2, "Accessories"), (3, "Footwear"), (4, "Outerwear")], format_func=lambda x: x[1])[0]
    
    Subscription_Status_encoded = st.selectbox("Subscription Status", [(0, "Inactive"), (1, "Active")], format_func=lambda x: x[1])[0]
    
    Preferred_Payment_Method_encoded = st.selectbox("Payment Method", [
        (1, "PayPal"), (2, "Credit Card"), (3, "Cash"), (4, "Debit Card"), (5, "Venmo"), (6, "Bank Transfer")
    ], format_func=lambda x: x[1])[0]

    Frequency_of_Purchases_encoded = st.selectbox("Frequency of Purchases", [
        (1, "Every 3 Months"), (2, "Annually"), (3, "Quarterly"), (4, "Monthly"),
        (5, "Bi-Weekly"), (6, "Fortnightly"), (7, "Weekly")
    ], format_func=lambda x: x[1])[0]

with col2:
    Purchase_Amount = st.number_input("Purchase Amount (USD)", min_value=0, max_value=10000, value=50)
    Previous_Purchases = st.number_input("Previous Purchases", min_value=0, max_value=100, value=5)
    Review_Rating = st.number_input("Customer Review Rating", min_value=1, max_value=5, value=3)
    
    Discount_Applied_encoded = st.selectbox("Discount Applied?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
    
    Promo_Code_Used_encoded = st.selectbox("Promo Code Used?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
    
    Shipping_Type_encoded = st.selectbox("Shipping Type", [
        (1, "Free Shipping"), (2, "Standard"), (3, "Store Pickup"),
        (4, "Next Day Air"), (5, "Express"), (6, "2-Day Shipping")
    ], format_func=lambda x: x[1])[0]

st.markdown("---")

# Prepare input data for models
clv_input = np.array([
    Gender_encoded, ItemPurchased_encoded, Category_encoded, 
    Subscription_Status_encoded, Preferred_Payment_Method_encoded, Purchase_Amount, 
    Previous_Purchases, Review_Rating
]).reshape(1, -1)

churn_input = np.array([
    Purchase_Amount, Previous_Purchases, Frequency_of_Purchases_encoded, Review_Rating
]).reshape(1, -1)

# Model Predictions
st.header("Predictions")

if st.button("Predict CLV", key="predict_clv"):
    expected_features_clv = clv_model.n_features_in_
    
    if clv_input.shape[1] != expected_features_clv:
        st.error(f"Feature mismatch: Model expects {expected_features_clv} features, but received {clv_input.shape[1]}.")
    else:
        clv_prediction = clv_model.predict(clv_input)[0]
        clv_label = "Below 1516" if clv_prediction == 0 else "1516 or Higher"

        st.markdown(f'<div class="result-box">Predicted CLV: <strong>{clv_label}</strong></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="result-box">Predicted CLV: <strong>{clv_prediction}</strong></div>', unsafe_allow_html=True)

if st.button("Predict Churn", key="predict_churn"):
    expected_features_churn = churn_model.n_features_in_
    if churn_input.shape[1] != expected_features_churn:
        st.error(f"Feature mismatch: Model expects {expected_features_churn} features, but received {churn_input.shape[1]}.")
    else:
        churn_prediction = churn_model.predict(churn_input)[0]
        churn_label = "Churned" if churn_prediction == 1 else "Not Churned"
        st.markdown(f'<div class="result-box">Predicted Churn Status: <strong>{churn_label}</strong></div>', unsafe_allow_html=True)

st.markdown("---")
st.write("**Recommendations:** Improve customer retention by offering discounts, improving engagement, and providing better payment options.")
