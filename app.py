import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Minimal Bond Calculator")
st.write("A minimal bond calculator app")

# Bond parameters
face_value = st.slider("Face Value ($)", 100, 10000, 1000, 100)
coupon_rate = st.slider("Coupon Rate (%)", 0.0, 20.0, 5.0, 0.1) / 100
years = st.slider("Years to Maturity", 1, 30, 10)
market_rate = st.slider("Market Rate (%)", 0.0, 20.0, 4.0, 0.1) / 100

# Simple bond price calculation
def calculate_bond_price(face_value, coupon_rate, years, market_rate):
    coupon_payment = face_value * coupon_rate
    periods = years
    
    # PV of coupon payments
    coupon_pv = 0
    for t in range(1, periods + 1):
        coupon_pv += coupon_payment / ((1 + market_rate) ** t)
    
    # PV of face value
    face_value_pv = face_value / ((1 + market_rate) ** periods)
    
    return coupon_pv + face_value_pv

# Calculate bond price
price = calculate_bond_price(face_value, coupon_rate, years, market_rate)

# Display results
st.write(f"Bond Price: ${price:.2f}")

# Simple chart
rates = np.linspace(max(0.01, market_rate - 0.05), market_rate + 0.05, 100)
prices = [calculate_bond_price(face_value, coupon_rate, years, r) for r in rates]

fig, ax = plt.subplots()
ax.plot(rates * 100, prices)
ax.set_xlabel("Market Rate (%)")
ax.set_ylabel("Bond Price ($)")
ax.set_title("Bond Price vs Market Rate")
ax.grid(True)

st.pyplot(fig)