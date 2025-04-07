import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Simple Bond Calculator",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Simple Bond Calculator")
st.write("A simple bond calculator for demonstration purposes")

# Sidebar inputs
st.sidebar.header("Bond Parameters")
face_value = st.sidebar.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100)
coupon_rate = st.sidebar.slider("Coupon Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1) / 100
years_to_maturity = st.sidebar.slider("Years to Maturity", min_value=1, max_value=30, value=10)
payments_per_year = st.sidebar.selectbox("Payments per Year", [1, 2, 4, 12], index=1)
market_rate = st.sidebar.slider("Market Interest Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.1) / 100

# Bond calculations
def calculate_bond_price(face_value, coupon_rate, years, periods, market_rate):
    # Calculate the bond price
    coupon_payment = face_value * coupon_rate / periods
    total_periods = years * periods
    discount_rate = market_rate / periods
    
    # Calculate the present value of coupon payments
    coupon_pv = coupon_payment * (1 - (1 + discount_rate) ** (-total_periods)) / discount_rate
    
    # Calculate the present value of the face value
    face_value_pv = face_value * (1 + discount_rate) ** (-total_periods)
    
    # Calculate the bond price
    bond_price = coupon_pv + face_value_pv
    
    return bond_price

def calculate_current_yield(bond_price, face_value, coupon_rate):
    annual_coupon = face_value * coupon_rate
    return annual_coupon / bond_price

def calculate_ytm(bond_price, face_value, coupon_rate, years, periods):
    # Estimate YTM using bisection method
    def npv(rate):
        return calculate_bond_price(face_value, coupon_rate, years, periods, rate) - bond_price
    
    # Initial bounds for bisection
    lower_bound = 0.0001
    upper_bound = 1.0
    
    # Bisection algorithm
    tolerance = 0.0001
    max_iterations = 100
    i = 0
    
    while i < max_iterations:
        mid_rate = (lower_bound + upper_bound) / 2
        npv_mid = npv(mid_rate)
        
        if abs(npv_mid) < tolerance:
            return mid_rate
        
        if npv_mid > 0:
            lower_bound = mid_rate
        else:
            upper_bound = mid_rate
        
        i += 1
    
    return (lower_bound + upper_bound) / 2

# Main calculations
bond_price = calculate_bond_price(face_value, coupon_rate, years_to_maturity, payments_per_year, market_rate)
current_yield = calculate_current_yield(bond_price, face_value, coupon_rate)
ytm = calculate_ytm(bond_price, face_value, coupon_rate, years_to_maturity, payments_per_year)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.header("Bond Valuation")
    st.metric("Bond Price", f"${bond_price:.2f}")
    st.metric("Current Yield", f"{current_yield*100:.2f}%")
    st.metric("Yield to Maturity", f"{ytm*100:.2f}%")
    
    # Show premium/discount status
    if bond_price > face_value:
        st.info(f"Bond is trading at a premium of ${bond_price - face_value:.2f}")
    elif bond_price < face_value:
        st.warning(f"Bond is trading at a discount of ${face_value - bond_price:.2f}")
    else:
        st.success("Bond is trading at par value")

with col2:
    # Create a chart for bond price vs market rate
    st.subheader("Bond Price vs Market Rate")
    
    rates = np.linspace(max(0.5, market_rate*100-3), market_rate*100+3, 100) / 100
    prices = [calculate_bond_price(face_value, coupon_rate, years_to_maturity, payments_per_year, r) for r in rates]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rates*100, prices, 'b-')
    ax.axhline(y=bond_price, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=market_rate*100, color='g', linestyle='--', alpha=0.7)
    ax.set_xlabel('Market Interest Rate (%)')
    ax.set_ylabel('Bond Price ($)')
    ax.set_title('Bond Price vs Market Interest Rate')
    ax.grid(True, alpha=0.3)
    
    # Add point annotation
    ax.plot(market_rate*100, bond_price, 'ro')
    ax.annotate(f'({market_rate*100:.1f}%, ${bond_price:.2f})', 
                xy=(market_rate*100, bond_price), 
                xytext=(market_rate*100+0.5, bond_price+20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    
    st.pyplot(fig)

# Additional bond information
st.header("Bond Details")
st.write(f"Coupon Payment: ${face_value * coupon_rate / payments_per_year:.2f} per period")
st.write(f"Number of Periods: {years_to_maturity * payments_per_year}")
st.write(f"Total Interest Paid over Life of Bond: ${face_value * coupon_rate * years_to_maturity:.2f}")

# Add a footer
st.markdown("---")
st.markdown("Bond Calculator | A simple demonstration app")