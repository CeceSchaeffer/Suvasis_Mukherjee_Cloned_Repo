import streamlit as st
import os
import math
import pandas as pd

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

# Function to create interactive charts using Streamlit's built-in chart functions
def create_streamlit_chart(x_data, y_data, x_label, y_label, title):
    chart_data = pd.DataFrame({
        x_label: x_data,
        y_label: y_data
    })
    return chart_data

# Load data from data folder
@st.cache_data
def load_financial_data():
    # Try multiple potential data paths to handle different deployment environments
    data_paths = [
        "data/datasets",                               # Local development path
        "/app/data/datasets",                          # Docker path
        "data",                                        # Streamlit Cloud path root
        "data/data/datasets",                          # Nested directory structure
        "../data/datasets",                            # Relative path
        "/mount/src/bond_analytics/data/datasets",     # Streamlit Cloud absolute path
        "/mount/src/bond_analytics/data/data/datasets" # Streamlit Cloud with nested structure
    ]
    
    data_dir = None
    # Find first valid data directory
    for path in data_paths:
        if os.path.exists(path):
            data_dir = path
            break
    
    # If no directory found, default to the local path
    if data_dir is None:
        data_dir = "data/datasets"
        st.warning(f"⚠️ No valid data directory found. Trying {data_dir}")
    else:
        st.write(f"Loading data from: {data_dir}")
    datasets = {}
    
    # Dictionary of files to load
    files_to_load = {
        'mk_maturity': 'mk.maturity.csv',
        'mk_zero2': 'mk.zero2.csv',
        'bond_prices': 'bondprices.txt',
        'zero_prices': 'ZeroPrices.txt',
        'yields': 'yields.txt',
        'treasury_yields': 'treasury_yields.txt',
        'stock_bond': 'Stock_Bond.csv'
    }
    
    try:
        # Try to load each file
        for key, filename in files_to_load.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                if filename.endswith('.txt'):
                    datasets[key] = pd.read_csv(filepath, sep='\s+')
                else:
                    datasets[key] = pd.read_csv(filepath, header=0)
        
        # Check if we loaded at least some of the files
        if len(datasets) > 0:
            st.success(f"✅ Successfully loaded {len(datasets)} datasets")
            # List the loaded datasets
            if len(datasets) < len(files_to_load):
                missing = [name for name, file in files_to_load.items() if name not in datasets]
                st.warning(f"⚠️ Could not find these datasets: {', '.join(missing)}")
            return datasets, True
        else:
            st.error("❌ Failed to load any datasets")
            # List all attempted files and paths
            for name, file in files_to_load.items():
                st.write(f"Tried to load: {os.path.join(data_dir, file)}")
            return {}, False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, False

# Get real yield curve data from loaded datasets
def get_real_yield_curve_data(datasets, date_index=5):
    """Get real yield curve data from mk.zero2 dataset for a specific date"""
    if 'mk_zero2' not in datasets or 'mk_maturity' not in datasets:
        return None, None, None
    
    # Get maturity values
    maturities = datasets['mk_maturity'].iloc[:, 0].values
    
    # Get yield values for the specified date
    date = datasets['mk_zero2'].iloc[date_index, 0]
    yields = datasets['mk_zero2'].iloc[date_index, 1:].values
    
    # Remove NaN values
    valid_indices = ~pd.isna(yields)
    maturities = maturities[valid_indices]
    yields = yields[valid_indices]
    
    return date, maturities, yields

# Get treasury yield curve data
def get_treasury_yield_curve(datasets, date_index=0):
    """Get treasury yield curve data for a specific date"""
    if 'treasury_yields' not in datasets:
        return None, None, None
    
    # Get the data for the specified date
    date = datasets['treasury_yields'].iloc[date_index, 0]
    
    # Extract maturities and yields
    # Convert column names to numeric values where possible
    maturities = []
    for col in datasets['treasury_yields'].columns[1:]:
        if col == '1mo':
            maturities.append(1/12)
        elif col == '3mo':
            maturities.append(3/12)
        elif col == '6mo':
            maturities.append(6/12)
        elif col == '1yr':
            maturities.append(1)
        elif col == '2yr':
            maturities.append(2)
        elif col == '3yr':
            maturities.append(3)
        elif col == '5yr':
            maturities.append(5)
        elif col == '7yr':
            maturities.append(7)
        elif col == '10yr':
            maturities.append(10)
        elif col == '20yr':
            maturities.append(20)
        elif col == '30yr':
            maturities.append(30)
    
    # Get yields for the date
    yields = datasets['treasury_yields'].iloc[date_index, 1:].values
    
    # Convert yields to decimal
    yields = yields / 100
    
    # Remove NaN values
    valid_indices = ~pd.isna(yields)
    maturities = [m for i, m in enumerate(maturities) if valid_indices[i]]
    yields = yields[valid_indices]
    
    return date, maturities, yields

# Get zero coupon bond data
def get_zero_coupon_bond_data(datasets):
    """Get zero coupon bond prices from the dataset"""
    if 'zero_prices' not in datasets:
        return None, None
    
    maturities = datasets['zero_prices']['maturity'].values
    prices = datasets['zero_prices']['price'].values
    
    # Calculate yields (continuously compounded)
    yields = [-math.log(p / 100) / m for p, m in zip(prices, maturities)]
    
    return maturities, prices, yields

# Load financial data
datasets, data_loaded = load_financial_data()

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Calculator", "Advanced Functions", "Real Data Analysis", "Educational Resources", "Chat Interface"])

with tab4:
    st.header("Educational Resources")
    
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
    st.header("Real Data Analysis")
    
    analysis_type = st.selectbox(
        "Select analysis type",
        ["Yield Curve Analysis", "Treasury Yield Analysis", "Zero-Coupon Bond Analysis"],
        key="real_data_analysis_type"
    )
    
    # Status message for data loading
    if not data_loaded:
        st.warning("⚠️ Could not load financial data. Some features might not work correctly.")
    
    if analysis_type == "Yield Curve Analysis":
        st.subheader("Yield Curve Analysis from Real Data")
        
        if data_loaded and 'mk_zero2' in datasets:
            # Get available dates
            dates = datasets['mk_zero2'].iloc[:, 0].values
            date_index = st.slider("Select Date", 0, len(dates)-1, 5, 1, key="yield_curve_date")
            selected_date = dates[date_index]
            
            # Get yield curve data
            date, maturities, yields = get_real_yield_curve_data(datasets, date_index)
            
            if date is not None:
                # Create a chart using Streamlit's built-in chart
                chart_data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Yield (%)': yields*100
                })
                
                st.line_chart(chart_data.set_index('Maturity (years)'))
                
                # Calculate forward rates
                forward_rates = []
                for i in range(1, len(yields)):
                    t1 = maturities[i-1]
                    t2 = maturities[i]
                    r1 = yields[i-1]
                    r2 = yields[i]
                    forward_rate = (r2*t2 - r1*t1)/(t2 - t1)
                    forward_rates.append(forward_rate)
                
                forward_maturities = [(maturities[i] + maturities[i-1])/2 for i in range(1, len(maturities))]
                
                # Create two separate charts for comparison
                st.write("### Spot Rates vs. Forward Rates")
                
                # Convert to DataFrames for Streamlit charts
                spot_rates_df = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Spot Rate (%)': yields*100
                })
                
                forward_rates_df = pd.DataFrame({
                    'Maturity (years)': forward_maturities,
                    'Forward Rate (%)': [r*100 for r in forward_rates]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Spot Rates")
                    st.line_chart(spot_rates_df.set_index('Maturity (years)'))
                    
                with col2:
                    st.write("Forward Rates")
                    st.line_chart(forward_rates_df.set_index('Maturity (years)'))
                
                # Display data as table
                data_table = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Spot Rate (%)': [y*100 for y in yields]
                })
                
                if len(forward_rates) > 0:
                    forward_df = pd.DataFrame({
                        'Period': [f"{maturities[i]:.1f}-{maturities[i+1]:.1f}" for i in range(len(forward_rates))],
                        'Forward Rate (%)': [r*100 for r in forward_rates]
                    })
                    
                    st.write("### Spot Rates")
                    st.dataframe(data_table)
                    
                    st.write("### Forward Rates")
                    st.dataframe(forward_df)
                else:
                    st.write("### Yield Curve Data")
                    st.dataframe(data_table)
            else:
                st.error("Could not retrieve yield curve data for the selected date.")
        else:
            st.error("Yield curve data files (mk.zero2.csv, mk.maturity.csv) not found.")
    
    elif analysis_type == "Treasury Yield Analysis":
        st.subheader("Treasury Yield Analysis from Real Data")
        
        if data_loaded and 'treasury_yields' in datasets:
            # Get available dates
            dates = datasets['treasury_yields'].iloc[:, 0].values
            date_index = st.slider("Select Date", 0, len(dates)-1, 0, 1, key="treasury_date")
            selected_date = dates[date_index]
            
            # Get treasury yield curve data
            date, maturities, yields = get_treasury_yield_curve(datasets, date_index)
            
            if date is not None:
                # Create a chart using Streamlit's built-in chart
                chart_data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Yield (%)': [y*100 for y in yields]
                })
                
                st.line_chart(chart_data.set_index('Maturity (years)'))
                
                # Display data as table
                data_table = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Yield (%)': [y*100 for y in yields]
                })
                
                st.write("### Treasury Yield Data")
                st.dataframe(data_table)
            else:
                st.error("Could not retrieve treasury yield data for the selected date.")
        else:
            st.error("Treasury yield data file (treasury_yields.txt) not found.")
    
    elif analysis_type == "Zero-Coupon Bond Analysis":
        st.subheader("Zero-Coupon Bond Analysis from Real Data")
        
        if data_loaded and 'zero_prices' in datasets:
            # Get zero coupon bond data
            maturities, prices, yields = get_zero_coupon_bond_data(datasets)
            
            if maturities is not None:
                # Create charts using Streamlit's built-in chart
                col1, col2 = st.columns(2)
                
                # Price chart
                price_data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Price': prices
                })
                
                # Yield chart
                yield_data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Yield (%)': [y*100 for y in yields]
                })
                
                with col1:
                    st.write("### Zero-Coupon Bond Prices")
                    st.line_chart(price_data.set_index('Maturity (years)'))
                
                with col2:
                    st.write("### Zero-Coupon Bond Yields")
                    st.line_chart(yield_data.set_index('Maturity (years)'))
                
                # Display data as table
                data_table = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Price': prices,
                    'Yield (%)': [y*100 for y in yields]
                })
                
                st.write("### Zero-Coupon Bond Data")
                st.dataframe(data_table)
                
                # Explanation
                st.write("""
                **Zero-Coupon Bonds** do not pay periodic interest but are sold at a discount to their face value. 
                The yield is the return an investor receives by holding the bond until maturity.
                
                The yield for a zero-coupon bond can be calculated as:
                
                Yield = -ln(Price/Face Value) / Maturity
                
                where the Price is typically quoted as a percentage of face value.
                """)
            else:
                st.error("Could not retrieve zero-coupon bond data.")
        else:
            st.error("Zero-coupon bond data file (ZeroPrices.txt) not found.")

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
        
    # Interactive Price vs Yield chart
    st.subheader("Price vs Yield Curve")
    
    # Generate yield values around the current yield
    min_yield = max(0.1, yield_rate * 100 - 2)
    max_yield = yield_rate * 100 + 2
    yields_for_plot = [y/100 for y in range(int(min_yield * 10), int(max_yield * 10) + 1)]
    prices_for_plot = [calculate_bond_price(face_value, coupon_rate, years, payments_per_year, y) for y in yields_for_plot]
    
    # Create DataFrame for Streamlit chart
    yield_price_data = pd.DataFrame({
        'Yield to Maturity (%)': [y * 100 for y in yields_for_plot],
        'Bond Price ($)': prices_for_plot
    })
    
    # Note: we can't add markers or annotations with Streamlit's built-in charts
    # But we can show the current values separately
    st.line_chart(yield_price_data.set_index('Yield to Maturity (%)'))
    
    # Show the current point
    st.info(f"Current point: Yield = {yield_rate*100:.2f}%, Price = ${price:.2f}")

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



with tab5:
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