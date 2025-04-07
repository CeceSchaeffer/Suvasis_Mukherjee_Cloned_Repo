import streamlit as st
import os
import math

st.set_page_config(
    page_title="Bond Calculator",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Bond Calculator")
st.write("A comprehensive bond calculator for financial engineering")

# Define common bond calculations that are used across tabs
def calculate_bond_price(face_value, coupon_rate, years, periods, yield_rate):
    """
    Computes bond values (current prices) corresponding to yield to maturity
    
    Parameters:
    face_value : float - par value of the bond
    coupon_rate : float - annual coupon rate as decimal (e.g., 0.05 for 5%)
    years : float - time to maturity (in years)
    periods : int - number of payments per year
    yield_rate : float - annual yield to maturity as decimal
    
    Returns:
    float - bond price
    """
    # Convert annual rates to per-period rates
    period_coupon_rate = coupon_rate / periods
    period_yield_rate = yield_rate / periods
    total_periods = years * periods
    
    # Calculate coupon payment per period
    coupon_payment = face_value * period_coupon_rate
    
    # Check for division by zero
    if period_yield_rate == 0:
        # For zero yield, it's just the sum of coupons plus par
        return coupon_payment * total_periods + face_value
    
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
    """Calculate the current yield of a bond"""
    annual_coupon = face_value * coupon_rate
    return annual_coupon / price

def calculate_bond_duration(face_value, coupon_rate, years, periods, yield_rate):
    """Calculate the Macaulay duration of a bond"""
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

def find_ytm_using_bisection(price, face_value, coupon_rate, years, periods, r_min=0.001, r_max=0.2, tolerance=0.0001, max_iterations=100):
    """Find yield to maturity using bisection method"""
    annual_coupon = face_value * coupon_rate
    period_coupon = annual_coupon / periods
    
    def bond_price_diff(r):
        calculated_price = calculate_bond_price(face_value, coupon_rate, years, periods, r)
        return calculated_price - price
    
    # Bisection algorithm
    lower_bound = r_min
    upper_bound = r_max
    i = 0
    
    while i < max_iterations:
        mid_rate = (lower_bound + upper_bound) / 2
        npv_mid = bond_price_diff(mid_rate)
        
        if abs(npv_mid) < tolerance:
            return mid_rate
        
        if npv_mid > 0:
            lower_bound = mid_rate
        else:
            upper_bound = mid_rate
        
        i += 1
    
    return (lower_bound + upper_bound) / 2

def calculate_spot_rate(forward_rate_func, T):
    """
    Calculate spot rate from forward rate function
    
    Parameters:
    forward_rate_func : function - the forward rate function r(t)
    T : float - time to maturity (in years)
    
    Returns:
    float - spot rate
    """
    if T <= 0:
        return forward_rate_func(0)
    
    # Numerically integrate the forward rate function
    num_steps = 1000
    dt = T / num_steps
    integral = 0
    
    for i in range(num_steps):
        t = i * dt
        integral += forward_rate_func(t) * dt
    
    return integral / T

def calculate_bond_price_with_spot_rates(coupon_payments, spot_rates, maturities, face_value=1000):
    """
    Calculate bond price using spot rates for each cash flow
    
    Parameters:
    coupon_payments : list - coupon payment amounts
    spot_rates : list - spot rates for each cash flow
    maturities : list - times to each cash flow
    face_value : float - par value of the bond
    
    Returns:
    float - bond price
    """
    price = 0
    for i, (payment, rate, maturity) in enumerate(zip(coupon_payments, spot_rates, maturities)):
        if i == len(coupon_payments) - 1:
            payment += face_value  # Add face value to final payment
        
        # Discount using the appropriate spot rate
        price += payment * math.exp(-rate * maturity)
    
    return price

def calculate_forward_rates(spot_rates, maturities):
    """
    Calculate forward rates from spot rates
    
    Parameters:
    spot_rates : list - spot rates for each maturity
    maturities : list - times to maturity
    
    Returns:
    list - forward rates between consecutive maturities
    """
    forward_rates = []
    for i in range(1, len(spot_rates)):
        t1 = maturities[i-1]
        t2 = maturities[i]
        r1 = spot_rates[i-1]
        r2 = spot_rates[i]
        
        # Calculate the forward rate between t1 and t2
        forward_rate = (r2 * t2 - r1 * t1) / (t2 - t1)
        forward_rates.append(forward_rate)
    
    return forward_rates

def find_coupon_given_ytm(price, ytm, years, periods, face_value):
    """
    Find the coupon payment given the price and yield to maturity
    
    Parameters:
    price : float - bond price
    ytm : float - yield to maturity (annual)
    years : float - time to maturity (in years)
    periods : int - number of payments per year
    face_value : float - par value of the bond
    
    Returns:
    float - annual coupon rate
    """
    period_ytm = ytm / periods
    total_periods = years * periods
    
    discount_factor = (1 + period_ytm) ** (-total_periods)
    period_coupon = period_ytm * (price - face_value * discount_factor) / (1 - discount_factor)
    annual_coupon_rate = period_coupon * periods / face_value
    
    return annual_coupon_rate

# Define tabs
tab1, tab2, tab3, tab4 = st.tabs(["Basic Calculator", "Advanced Functions", "Educational Resources", "Chat Interface"])

with tab2:
    st.header("Advanced Bond Functions")
    st.sidebar.header("Advanced Parameters")
    
    advanced_function = st.sidebar.selectbox(
        "Select Function",
        ["Forward Rate & Spot Rate", "Pricing with Spot Rates", "Find YTM", "Find Coupon Given YTM"]
    )
    
    if advanced_function == "Forward Rate & Spot Rate":
        st.subheader("Forward Rate to Spot Rate Calculator")
        
        # Define a forward rate function
        forward_rate_type = st.selectbox(
            "Forward Rate Function Type",
            ["Constant", "Linear", "Quadratic"]
        )
        
        if forward_rate_type == "Constant":
            constant_rate = st.slider("Constant Rate (%)", 0.0, 15.0, 2.8, 0.1) / 100
            st.write(f"Forward Rate Function: r(t) = {constant_rate*100:.1f}%")
            
            def forward_rate_func(t):
                return constant_rate
                
        elif forward_rate_type == "Linear":
            a_coef = st.slider("Constant Term (a) (%)", 0.0, 10.0, 2.8, 0.1) / 100
            b_coef = st.slider("Slope (b) (%/year)", 0.0, 2.0, 0.042, 0.001) / 100
            st.write(f"Forward Rate Function: r(t) = {a_coef*100:.1f}% + {b_coef*100:.3f}% × t")
            
            def forward_rate_func(t):
                return a_coef + b_coef * t
                
        elif forward_rate_type == "Quadratic":
            a_coef = st.slider("Constant Term (a) (%)", 0.0, 10.0, 2.8, 0.1) / 100
            b_coef = st.slider("Linear Term (b) (%/year)", 0.0, 2.0, 0.042, 0.001) / 100
            c_coef = st.slider("Quadratic Term (c) (%/year²)", -0.1, 0.1, 0.0, 0.001) / 100
            st.write(f"Forward Rate Function: r(t) = {a_coef*100:.1f}% + {b_coef*100:.3f}% × t + {c_coef*100:.3f}% × t²")
            
            def forward_rate_func(t):
                return a_coef + b_coef * t + c_coef * t * t
        
        # Calculate spot rates for different maturities
        maturities = [1, 2, 3, 5, 7, 10, 15, 20, 30]
        spot_rates = [calculate_spot_rate(forward_rate_func, T) for T in maturities]
        
        # Display results in a table
        results = []
        for T, spot_rate in zip(maturities, spot_rates):
            forward_rate = forward_rate_func(T)
            results.append({
                "Maturity (years)": T,
                "Forward Rate (%)": f"{forward_rate*100:.4f}%",
                "Spot Rate (%)": f"{spot_rate*100:.4f}%"
            })
        
        st.table(results)
        
        st.write("""
        **Forward Rate** represents the instantaneous interest rate at time t.
        
        **Spot Rate** (or yield to maturity) for a zero-coupon bond maturing at time T is the average of forward rates from 0 to T:
        
        R(T) = (1/T) ∫₀ᵀ r(t) dt
        
        where r(t) is the forward rate function.
        """)
        
    elif advanced_function == "Pricing with Spot Rates":
        st.subheader("Bond Pricing with Spot Rates")
        
        # Bond parameters
        face_value = st.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100, key="spot_face_value")
        coupon_rate = st.slider("Annual Coupon Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="spot_coupon_rate") / 100
        years = st.slider("Years to Maturity", min_value=1, max_value=10, value=5, key="spot_years")
        payments_per_year = st.selectbox("Payments per Year", [1, 2, 4, 12], index=1, key="spot_payments_per_year")
        
        st.write("### Spot Rate Term Structure")
        st.write("Enter spot rates for each cash flow (these are typically different from a single YTM):")
        
        # Generate times for cash flows
        total_payments = int(years * payments_per_year)
        times = [t/payments_per_year for t in range(1, total_payments+1)]
        
        # Input spot rates for each time
        spot_rates = []
        col1, col2 = st.columns(2)
        
        for i, t in enumerate(times):
            if i % 2 == 0:
                with col1:
                    rate = st.slider(f"Spot Rate at t={t:.2f} years (%)", 0.0, 15.0, 
                                    min(4.0 + t*0.2, 15.0), 0.1, key=f"spot_rate_{i}") / 100
            else:
                with col2:
                    rate = st.slider(f"Spot Rate at t={t:.2f} years (%)", 0.0, 15.0, 
                                    min(4.0 + t*0.2, 15.0), 0.1, key=f"spot_rate_{i}") / 100
            spot_rates.append(rate)
        
        # Calculate coupon payments
        period_coupon = face_value * coupon_rate / payments_per_year
        coupon_payments = [period_coupon] * total_payments
        
        # Calculate bond price using spot rates
        price = calculate_bond_price_with_spot_rates(coupon_payments, spot_rates, times, face_value)
        
        # Display the result
        st.success(f"Bond Price: ${price:.2f}")
        
        # Calculate implied YTM for comparison
        ytm = find_ytm_using_bisection(price, face_value, coupon_rate, years, payments_per_year)
        st.info(f"Implied Yield to Maturity: {ytm*100:.4f}%")
        
        # Display explanatory text
        st.write("""
        This calculator prices a bond using different spot rates for each cash flow, rather than a single yield to maturity.
        
        The price formula is:
        
        Price = Σ(CFₜ × e^(-r₍ₜ₎ × t))
        
        where:
        - CFₜ is the cash flow at time t
        - r₍ₜ₎ is the spot rate for maturity t
        - t is the time of the cash flow
        
        This approach more accurately reflects the term structure of interest rates.
        """)
        
    elif advanced_function == "Find YTM":
        st.subheader("Find Yield to Maturity (YTM)")
        
        # Input parameters
        face_value = st.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100, key="ytm_face_value")
        price = st.number_input("Bond Price ($)", min_value=10.0, max_value=10000.0, value=950.0, step=10.0, key="ytm_price")
        coupon_rate = st.slider("Annual Coupon Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="ytm_coupon_rate") / 100
        years = st.slider("Years to Maturity", min_value=1, max_value=30, value=10, key="ytm_years")
        payments_per_year = st.selectbox("Payments per Year", [1, 2, 4, 12], index=1, key="ytm_payments_per_year")
        
        # Find YTM
        ytm = find_ytm_using_bisection(price, face_value, coupon_rate, years, payments_per_year)
        
        # Display results
        st.success(f"Yield to Maturity: {ytm*100:.4f}%")
        
        # Calculate current yield for comparison
        current_yield = (face_value * coupon_rate) / price
        st.info(f"Current Yield: {current_yield*100:.4f}%")
        
        # Calculate bond status
        if price > face_value:
            st.write(f"Bond is trading at a premium of ${price - face_value:.2f}")
        elif price < face_value:
            st.write(f"Bond is trading at a discount of ${face_value - price:.2f}")
        else:
            st.write("Bond is trading at par value")
            
        st.write("""
        **Yield to Maturity (YTM)** is the total return anticipated on a bond if held until maturity.
        It's calculated by finding the interest rate that makes the present value of all future cash flows
        equal to the current price of the bond.
        
        This calculator uses a bisection algorithm to find the YTM.
        """)
        
    elif advanced_function == "Find Coupon Given YTM":
        st.subheader("Find Coupon Rate Given Price and YTM")
        
        # Input parameters
        face_value = st.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100, key="coupon_face_value")
        price = st.number_input("Bond Price ($)", min_value=10.0, max_value=10000.0, value=1000.0, step=10.0, key="coupon_price")
        ytm = st.slider("Yield to Maturity (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1, key="coupon_ytm") / 100
        years = st.slider("Years to Maturity", min_value=1, max_value=30, value=10, key="coupon_years")
        payments_per_year = st.selectbox("Payments per Year", [1, 2, 4, 12], index=1, key="coupon_payments_per_year")
        
        # Find coupon rate
        coupon_rate = find_coupon_given_ytm(price, ytm, years, payments_per_year, face_value)
        
        # Display results
        st.success(f"Required Annual Coupon Rate: {coupon_rate*100:.4f}%")
        st.info(f"Coupon Payment Per Period: ${face_value * coupon_rate / payments_per_year:.2f}")
        st.info(f"Total Annual Coupon: ${face_value * coupon_rate:.2f}")
        
        st.write("""
        This calculator determines the coupon rate needed for a bond to trade at a specific price given its yield to maturity.
        
        This is useful for:
        - Designing new bond issues
        - Understanding the relationship between coupon rates, prices, and yields
        - Analyzing bond market dynamics
        """)

with tab1:
    st.header("Basic Bond Calculator")
    st.sidebar.header("Basic Parameters")

    # Bond parameters inputs
    face_value = st.sidebar.number_input("Face Value ($)", min_value=100, max_value=10000, value=1000, step=100, key="basic_face_value")
    coupon_rate = st.sidebar.slider("Annual Coupon Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="basic_coupon_rate") / 100
    years = st.sidebar.slider("Years to Maturity", min_value=1, max_value=30, value=10, key="basic_years")
    payments_per_year = st.sidebar.selectbox("Payments per Year", [1, 2, 4, 12], index=1, key="basic_payments_per_year")
    yield_rate = st.sidebar.slider("Yield to Maturity (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1, key="basic_yield_rate") / 100

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

with tab2:
    st.header("Bond Calculation Concepts")
    
    with st.expander("Yield to Maturity (YTM)"):
        st.write("""
        **Yield to Maturity (YTM)** is the total return anticipated on a bond if held until it matures. It's calculated by 
        finding the interest rate that makes the present value of all future cash flows equal to the current price of the bond.
        
        For a bond with semiannual coupon payments, the formula is:
        
        Price = C/r + (Par - C/r) * (1 + r)^(-2T)
        
        Where:
        - C = semiannual coupon payment
        - r = semiannual yield to maturity
        - T = time to maturity in years
        - Par = par value of the bond
        """)
    
    with st.expander("Bond Duration"):
        st.write("""
        **Bond Duration** measures the sensitivity of a bond's price to interest rate changes. It represents the weighted 
        average time to receive the bond's cash flows.
        
        Macaulay Duration is calculated as:
        
        Duration = Σ(t × PV(CFt)) / Price
        
        Where:
        - t = time to each cash flow
        - PV(CFt) = present value of the cash flow at time t
        - Price = bond price
        
        **Modified Duration** adjusts Macaulay Duration to estimate the percentage price change for a 1% change in yield:
        
        Modified Duration = Macaulay Duration / (1 + YTM/n)
        
        Where n is the number of coupon payments per year.
        """)
    
    with st.expander("Forward Rates"):
        st.write("""
        **Forward Rates** represent the interest rate for a future period that is implied by the current term structure of interest rates.
        
        The forward rate between times t₁ and t₂ is calculated as:
        
        Forward Rate(t₁,t₂) = (Spot Rate(t₂) × t₂ - Spot Rate(t₁) × t₁) / (t₂ - t₁)
        
        Where Spot Rate(t) is the yield to maturity of a zero-coupon bond maturing at time t.
        """)
    
    with st.expander("Yield Curve"):
        st.write("""
        A **Yield Curve** shows the relationship between interest rates (or yields) and time to maturity for debt securities 
        with equal credit quality but different maturity dates.
        
        The shape of the yield curve can indicate economic forecasts and expectations of future interest rate changes:
        1. **Normal (Upward Sloping)**: Long-term yields higher than short-term yields, indicating economic expansion
        2. **Inverted (Downward Sloping)**: Short-term yields higher than long-term yields, often predicting recession
        3. **Flat**: Similar yields for short and long-term maturities
        4. **Humped**: Medium-term yields higher than both short and long-term yields
        """)
    
    with st.expander("Zero-Coupon Bonds"):
        st.write("""
        **Zero-Coupon Bonds** don't pay periodic interest but are sold at a discount to their face value. The investor's return
        comes from the difference between the purchase price and the face value at maturity.
        
        The price of a zero-coupon bond is:
        
        Price = Face Value / (1 + YTM)^t
        
        Where t is the time to maturity in years.
        
        The yield of a zero-coupon bond can also be calculated using continuous compounding:
        
        Yield = -ln(Price/Face Value) / t
        """)

with tab3:
    st.header("Chat with the Bond Calculator")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Bond Calculator Chat! I can help you with bond calculations, explain bond concepts, or provide information about financial data. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What would you like to know about bonds?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Generate response based on predefined answers
        response = ""
        prompt_lower = prompt.lower()
        
        # Process the prompt and generate response
        if "yield to maturity" in prompt_lower or "ytm" in prompt_lower:
            response = """
            **Yield to Maturity (YTM)** is the total return anticipated on a bond if held until it matures. It's calculated by finding the interest rate that makes the present value of all future cash flows equal to the current price of the bond.
            
            For a bond with semiannual coupon payments, the formula is:
            
            Price = C/r + (Par - C/r) * (1 + r)^(-2T)
            
            Where:
            - C = semiannual coupon payment
            - r = semiannual yield to maturity
            - T = time to maturity in years
            - Par = par value of the bond
            
            To calculate YTM for a specific bond, you can use the calculator tab.
            """
        
        elif "duration" in prompt_lower:
            response = """
            **Bond Duration** measures the sensitivity of a bond's price to interest rate changes. It represents the weighted average time to receive the bond's cash flows.
            
            There are two main types:
            
            1. **Macaulay Duration**: The weighted average time to receive cash flows, measured in years.
            
            2. **Modified Duration**: Estimates the percentage price change for a 1% change in yield.
            
            The formula for Macaulay Duration is:
            Duration = Σ(t × PV(CFt)) / Price
            
            Modified Duration = Macaulay Duration / (1 + YTM/n)
            
            Where n is the number of coupon payments per year.
            
            Higher duration means higher interest rate risk.
            """
        
        elif "forward rate" in prompt_lower:
            response = """
            **Forward Rates** represent future interest rates implied by the current term structure of interest rates. They can be derived from spot rates (yields).
            
            The forward rate between times t₁ and t₂ is calculated from spot rates using:
            
            Forward Rate(t₁,t₂) = (Spot Rate(t₂) × t₂ - Spot Rate(t₁) × t₁) / (t₂ - t₁)
            
            Forward rates are important for:
            - Forecasting future interest rates
            - Pricing fixed income derivatives
            - Managing interest rate risk
            - Analyzing the yield curve
            """
        
        elif "yield curve" in prompt_lower:
            response = """
            A **Yield Curve** shows the relationship between interest rates (yields) and time to maturity for debt securities with equal credit quality.
            
            The shape of the yield curve can indicate economic forecasts:
            
            1. **Normal (Upward Sloping)**: Long-term yields > short-term yields, indicating economic expansion
            2. **Inverted (Downward Sloping)**: Short-term yields > long-term yields, often predicting recession
            3. **Flat**: Similar yields for short and long-term maturities
            4. **Humped**: Medium-term yields higher than both short and long-term yields
            
            The yield curve is used for:
            - Economic forecasting
            - Pricing bonds
            - Developing trading strategies
            - Risk management
            """
        
        elif "zero coupon" in prompt_lower or "zero-coupon" in prompt_lower:
            response = """
            **Zero-Coupon Bonds** don't pay periodic interest but are sold at a discount to their face value. The investor's return comes from the difference between the purchase price and the face value at maturity.
            
            The price of a zero-coupon bond is:
            
            Price = Face Value / (1 + YTM)^t
            
            Where t is the time to maturity in years.
            
            The yield can be calculated using continuous compounding:
            
            Yield = -ln(Price/Face Value) / t
            
            Zero-coupon bonds:
            - Have no reinvestment risk
            - Have higher price volatility (higher duration)
            - Are often used for targeted financial planning (education, retirement)
            - Can have tax implications as imputed interest
            """
        
        elif "calculate" in prompt_lower and "price" in prompt_lower:
            response = """
            To calculate bond price, use the calculator in the first tab.
            
            The basic formula for bond price is:
            
            Price = Σ(Coupon Payment / (1 + YTM)^t) + Face Value / (1 + YTM)^n
            
            Where:
            - Coupon Payment is the periodic interest payment
            - YTM is the yield to maturity per period
            - t is the time of each payment
            - n is the total number of periods
            
            The calculator handles all these calculations automatically when you input the bond parameters.
            """
        
        elif "current yield" in prompt_lower:
            response = """
            **Current Yield** is the annual interest payment divided by the current market price of the bond.
            
            Current Yield = (Annual Coupon Payment / Current Bond Price) × 100%
            
            Current yield differs from yield to maturity in that:
            - It only considers the coupon payments, not capital gains/losses
            - It doesn't account for the time value of money
            - It doesn't consider the bond's remaining time to maturity
            
            Current yield is useful for comparing bond income to other income-generating investments.
            """
        
        elif "premium" in prompt_lower or "discount" in prompt_lower:
            response = """
            Bonds can trade at premium, discount, or par:
            
            - **Premium Bond**: Price > Face Value. Occurs when the bond's coupon rate > market yield.
            - **Discount Bond**: Price < Face Value. Occurs when the bond's coupon rate < market yield.
            - **Par Bond**: Price = Face Value. Occurs when the bond's coupon rate = market yield.
            
            For a premium bond:
            - YTM < Coupon Rate
            - Current Yield < Coupon Rate
            
            For a discount bond:
            - YTM > Coupon Rate
            - Current Yield > Coupon Rate
            
            Premium bonds have less interest rate risk (lower duration) than discount bonds of the same maturity.
            """
        
        elif "random walk" in prompt_lower or "stock model" in prompt_lower:
            response = """
            The **Random Walk Model** is used in finance to model stock prices and other financial assets.
            
            In its continuous form, it's known as **Geometric Brownian Motion**:
            
            dS = μS dt + σS dW
            
            Where:
            - S is the stock price
            - μ is the drift (expected return)
            - σ is the volatility
            - dW is a random increment from a normal distribution
            
            This model:
            - Forms the foundation for the Black-Scholes option pricing model
            - Assumes log-normal distribution of stock prices
            - Implies that price changes are unpredictable
            - Is consistent with the Efficient Market Hypothesis
            """
        
        elif "help" in prompt_lower:
            response = """
            I can help with various bond concepts and calculations including:
            
            - Yield to Maturity (YTM)
            - Bond Duration
            - Forward Rates
            - Yield Curve
            - Zero-Coupon Bonds
            - Bond Pricing
            - Current Yield
            - Premium/Discount Bonds
            - Random Walk Model
            
            You can ask specific questions like:
            - "What is yield to maturity?"
            - "How is bond duration calculated?"
            - "Explain forward rates"
            - "What does the yield curve tell us?"
            
            You can also use the calculator tab to perform bond calculations with your own parameters.
            """
        
        else:
            response = """
            I'm a bond calculator assistant that can help with bond concepts and calculations.
            
            Some topics I can discuss:
            - Yield to Maturity (YTM)
            - Bond Duration
            - Forward Rates
            - Yield Curve
            - Zero-Coupon Bonds
            - Bond Pricing
            - Current Yield
            - Premium/Discount Bonds
            
            Try asking a specific question about one of these topics, or use the calculator tab to perform bond calculations.
            """
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Footer
st.markdown("---")
st.caption("Bond Calculator | Financial Engineering Tools")