import streamlit as st

st.set_page_config(
    page_title="Minimal Bond Calculator",
    page_icon="💰",
    layout="wide"
)

st.title("Bond Calculator")
st.write("A simple bond calculator for financial engineering")

st.sidebar.header("Input Parameters")

# Bond parameters inputs
face_value = st.sidebar.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100)
coupon_rate = st.sidebar.slider("Annual Coupon Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
years = st.sidebar.slider("Years to Maturity", min_value=1, max_value=30, value=10)
payments_per_year = st.sidebar.selectbox("Payments per Year", [1, 2, 4, 12], index=1)
yield_rate = st.sidebar.slider("Yield to Maturity (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1) / 100

# Simple bond calculations
def calculate_bond_price(face_value, coupon_rate, years, periods, yield_rate):
    # Convert annual rates to per-period rates
    period_coupon_rate = coupon_rate / periods
    period_yield_rate = yield_rate / periods
    total_periods = years * periods
    
    # Calculate coupon payment per period
    coupon_payment = face_value * period_coupon_rate
    
    # Calculate present value of coupon payments
    coupon_pv = 0
    for i in range(1, int(total_periods) + 1):
        coupon_pv += coupon_payment / ((1 + period_yield_rate) ** i)
    
    # Calculate present value of face value
    face_value_pv = face_value / ((1 + period_yield_rate) ** total_periods)
    
    # Total bond price
    bond_price = coupon_pv + face_value_pv
    
    return bond_price

def calculate_current_yield(price, face_value, coupon_rate):
    annual_coupon = face_value * coupon_rate
    return annual_coupon / price

def calculate_bond_duration(face_value, coupon_rate, years, periods, yield_rate):
    period_coupon_rate = coupon_rate / periods
    period_yield_rate = yield_rate / periods
    total_periods = years * periods
    
    coupon_payment = face_value * period_coupon_rate
    
    weighted_pv_sum = 0
    pv_sum = 0
    
    for i in range(1, int(total_periods) + 1):
        period_in_years = i / periods
        
        if i < total_periods:
            cash_flow = coupon_payment
        else:
            cash_flow = coupon_payment + face_value
            
        pv = cash_flow / ((1 + period_yield_rate) ** i)
        weighted_pv_sum += period_in_years * pv
        pv_sum += pv
        
    duration = weighted_pv_sum / pv_sum
    return duration

# Calculate results
price = calculate_bond_price(face_value, coupon_rate, years, payments_per_year, yield_rate)
current_yield = calculate_current_yield(price, face_value, coupon_rate)
duration = calculate_bond_duration(face_value, coupon_rate, years, payments_per_year, yield_rate)
modified_duration = duration / (1 + yield_rate / payments_per_year)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.header("Bond Valuation")
    st.metric("Bond Price", f"${price:.2f}")
    st.metric("Current Yield", f"{current_yield*100:.2f}%")
    
    if price > face_value:
        st.info(f"Bond is trading at a premium of ${price - face_value:.2f}")
    elif price < face_value:
        st.warning(f"Bond is trading at a discount of ${face_value - price:.2f}")
    else:
        st.success("Bond is trading at par value")

with col2:
    st.header("Risk Metrics")
    st.metric("Duration", f"{duration:.2f} years")
    st.metric("Modified Duration", f"{modified_duration:.2f}")
    
    price_change = -modified_duration * 0.01 * price
    st.write(f"If yield increases by 1%, bond price changes by approximately ${price_change:.2f}")

# Bond details
st.header("Bond Cash Flows")

# Create a table of cash flows
cash_flows = []
for i in range(1, int(years * payments_per_year) + 1):
    period = i
    year = i / payments_per_year
    payment = face_value * (coupon_rate / payments_per_year)
    
    if i == years * payments_per_year:  # Last payment
        payment += face_value
        
    present_value = payment / ((1 + yield_rate / payments_per_year) ** i)
    
    cash_flows.append({
        "Period": period,
        "Year": f"{year:.2f}",
        "Payment": f"${payment:.2f}",
        "Present Value": f"${present_value:.2f}"
    })

# Display cash flows table
st.table(cash_flows)

st.markdown("---")
st.caption("Bond Calculator | A simple demonstration app")