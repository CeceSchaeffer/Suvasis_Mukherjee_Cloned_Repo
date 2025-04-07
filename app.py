import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from scipy.optimize import brentq
import os
import base64
from io import BytesIO
import pandas as pd
import datetime

class BondCalculator:
    """
    A comprehensive bond calculator that implements solutions for various bond-related problems
    using real financial data.
    """
    
    def __init__(self, data_dir="data/datasets"):
        """Initialize the BondCalculator class with data directory"""
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dir = data_dir
        self.data_loaded = False
        self.datasets = {}
        self.load_data()
    
    def load_data(self):
        """Load all relevant financial datasets"""
        try:
            # Load yield curve data
            self.datasets['mk_maturity'] = pd.read_csv(os.path.join(self.data_dir, 'mk.maturity.csv'), header=0)
            self.datasets['mk_zero2'] = pd.read_csv(os.path.join(self.data_dir, 'mk.zero2.csv'), header=0)
            
            # Load bond prices data
            self.datasets['bond_prices'] = pd.read_csv(os.path.join(self.data_dir, 'bondprices.txt'), sep='\s+')
            self.datasets['zero_prices'] = pd.read_csv(os.path.join(self.data_dir, 'ZeroPrices.txt'), sep='\s+')
            
            # Load yields data
            self.datasets['yields'] = pd.read_csv(os.path.join(self.data_dir, 'yields.txt'), sep='\s+')
            
            # Load treasury yields data
            self.datasets['treasury_yields'] = pd.read_csv(os.path.join(self.data_dir, 'treasury_yields.txt'), sep='\s+')
            
            # Load stock and bond data for correlation analysis
            self.datasets['stock_bond'] = pd.read_csv(os.path.join(self.data_dir, 'Stock_Bond.csv'), header=0)
            
            self.data_loaded = True
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def bondvalue(self, c, T, r, par=1000):
        """
        Computes bond values (current prices) corresponding to all values of yield to maturity
        
        Parameters:
        c : float - coupon payment (semiannual)
        T : float - time to maturity (in years)
        r : array or float - yields to maturity (semiannual rates)
        par : float - par value
        
        Returns:
        bv : array or float - bond values
        """
        bv = c / r + (par - c / r) * (1 + r) ** (-2 * T)
        return bv
    
    def calculate_spot_rate(self, forward_rate_func, T):
        """Calculate spot rate from forward rate function"""
        if T <= 0:
            return forward_rate_func(0)
        integral, _ = integrate.quad(forward_rate_func, 0, T)
        return integral / T
    
    def find_ytm_using_root(self, price, c, T, par, r_min=0.001, r_max=0.2):
        """Find yield to maturity using root-finding method"""
        def bond_price_diff(r, price, c, T, par):
            calculated_price = self.bondvalue(c, T, r, par)
            return calculated_price - price
        
        ytm = brentq(bond_price_diff, r_min, r_max, args=(price, c, T, par))
        return ytm
    
    def find_coupon_given_ytm(self, price, ytm, T, par):
        """Find the coupon payment given the yield to maturity"""
        discount_factor = (1 + ytm) ** (-2 * T)
        c = ytm * (price - par * discount_factor) / (1 - discount_factor)
        return c
    
    def calculate_bond_duration(self, c, T, ytm, par=1000):
        """Calculate the duration of a bond"""
        # For a coupon bond with semiannual payments
        r = ytm
        duration = 0
        total_npv = 0
        
        # Calculate NPV of each cash flow
        for i in range(1, int(2*T) + 1):
            t = i / 2  # Time in years
            if i < int(2*T):
                cash_flow = c
            else:
                cash_flow = c + par
            
            npv = cash_flow * (1 + r) ** (-i)
            total_npv += npv
            duration += t * npv
        
        # Duration is the weighted average time
        duration = duration / total_npv
        return duration
    
    def calculate_bond_price_with_spot_rates(self, coupon_payments, spot_rates, maturities, par=1000):
        """Calculate bond price using spot rates for each cash flow"""
        price = 0
        for i, (payment, rate, maturity) in enumerate(zip(coupon_payments, spot_rates, maturities)):
            if i == len(coupon_payments) - 1:
                payment += par  # Add par value to final payment
            
            # Discount using the appropriate spot rate
            price += payment * np.exp(-rate * maturity)
        
        return price
    
    def calculate_forward_rates(self, spot_rates, maturities):
        """Calculate forward rates from spot rates"""
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
    
    def get_real_yield_curve_data(self, date_index=5):
        """Get real yield curve data from mk.zero2 dataset for a specific date"""
        if not self.data_loaded:
            return None, None, None
        
        # Get maturity values
        maturities = self.datasets['mk_maturity'].iloc[:, 0].values
        
        # Get yield values for the specified date
        date = self.datasets['mk_zero2'].iloc[date_index, 0]
        yields = self.datasets['mk_zero2'].iloc[date_index, 1:].values
        
        # Remove NaN values
        valid_indices = ~np.isnan(yields)
        maturities = maturities[valid_indices]
        yields = yields[valid_indices]
        
        return date, maturities, yields
    
    def get_treasury_yield_curve(self, date_index=0):
        """Get treasury yield curve data for a specific date"""
        if not self.data_loaded:
            return None, None, None
        
        # Get the data for the specified date
        date = self.datasets['treasury_yields'].iloc[date_index, 0]
        
        # Extract maturities and yields
        # Convert column names to numeric values where possible
        maturities = []
        for col in self.datasets['treasury_yields'].columns[1:]:
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
        yields = self.datasets['treasury_yields'].iloc[date_index, 1:].values
        
        # Convert yields to decimal
        yields = yields / 100
        
        # Remove NaN values
        valid_indices = ~np.isnan(yields)
        maturities = np.array(maturities)[valid_indices]
        yields = yields[valid_indices]
        
        return date, maturities, yields
    
    def get_zero_coupon_bond_prices(self):
        """Get zero coupon bond prices from the dataset"""
        if not self.data_loaded:
            return None, None
        
        maturities = self.datasets['zero_prices']['maturity'].values
        prices = self.datasets['zero_prices']['price'].values
        
        return maturities, prices
    
    def get_bond_prices(self):
        """Get bond prices from the dataset"""
        if not self.data_loaded:
            return None, None
        
        maturities = self.datasets['bond_prices']['maturity'].values
        prices = self.datasets['bond_prices']['price'].values
        
        return maturities, prices
    
    def calculate_yields_from_zero_prices(self):
        """Calculate yields from zero coupon bond prices"""
        maturities, prices = self.get_zero_coupon_bond_prices()
        
        if maturities is None:
            return None, None
        
        # Calculate yields (continuously compounded)
        yields = -np.log(prices / 100) / maturities
        
        return maturities, yields
    
    def get_stock_bond_correlation(self, stock1='GM', stock2='Ford'):
        """Calculate correlation between two stocks from the Stock_Bond dataset"""
        if not self.data_loaded or 'stock_bond' not in self.datasets:
            return None, None, None
        
        # Get the data
        data = self.datasets['stock_bond']
        
        # Calculate log returns
        if stock1 in data.columns and stock2 in data.columns:
            stock1_returns = np.diff(np.log(data[stock1].values))
            stock2_returns = np.diff(np.log(data[stock2].values))
            
            # Calculate correlation
            correlation = np.corrcoef(stock1_returns, stock2_returns)[0, 1]
            
            return stock1_returns, stock2_returns, correlation
        
        return None, None, None
    
    def simulate_random_walk(self, S0, mu, sigma, T, dt=0.01, num_paths=5):
        """Simulate geometric Brownian motion paths"""
        num_steps = int(T / dt)
        times = np.linspace(0, T, num_steps)
        paths = np.zeros((num_paths, num_steps))
        paths[:, 0] = S0
        
        for i in range(num_paths):
            for t in range(1, num_steps):
                # Generate random normal increment
                dW = np.random.normal(0, np.sqrt(dt))
                # Update price using geometric Brownian motion
                paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
        return times, paths
    
    def get_plot_as_base64(self, fig):
        """Convert matplotlib figure to base64 string for embedding in HTML"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str
    
    def save_plot(self, fig, filename):
        """Save matplotlib figure to file"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=100)
        plt.close(fig)
        return filepath

# Initialize the bond calculator
bond_calculator = BondCalculator()

# Set up the Streamlit app
st.set_page_config(
    page_title="Bond Calculator Chat UI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: #fff;
    }
    .chat-message.bot {
        background-color: #475063;
        color: #fff;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2E4A9A;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .formula {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-family: monospace;
    }
    .question-card {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E3A8A;
    }
    .tabs-container {
        margin-top: 1rem;
    }
    .tab-content {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0 0.5rem 0.5rem 0.5rem;
    }
    .data-source {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Bond Calculator")
st.sidebar.markdown("---")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main app title
st.title("Bond Calculator Chat Interface")
st.markdown("Ask questions about bond calculations or select a specific problem to solve.")

# Question categories in the sidebar
st.sidebar.header("Question Categories")
category = st.sidebar.radio(
    "Select a category:",
    ["Chat Interface", "Real Data Analysis", "Bond Yield & Pricing (Q1-5)", 
     "Yield to Maturity (Q6-8)", "Bond Calculations (Q9-13)", "Advanced Analysis (Q14-20)"]
)

if category == "Chat Interface":
    # Chat interface
    st.markdown("### Chat with the Bond Calculator")
    st.markdown("Ask questions like 'What is yield to maturity?' or 'Show me real yield curve data'")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y">
                </div>
                <div class="message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://www.gravatar.com/avatar/00000000000000000000000000000000?d=identicon&f=y">
                </div>
                <div class="message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input("Your question:", key="user_input")
    
    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process the user input and generate a response
            if "yield curve" in user_input.lower() or "yield data" in user_input.lower():
                # Get real yield curve data
                date, maturities, yields = bond_calculator.get_real_yield_curve_data(date_index=5)
                
                if date is not None:
                    # Create a plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(maturities, yields, 'b-o', label=f'Yield Curve on {date}')
                    ax.set_xlabel('Maturity (years)')
                    ax.set_ylabel('Yield')
                    ax.set_title('Real Yield Curve Data')
                    ax.grid(True)
                    ax.legend()
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Real Yield Curve Data</h3>
                        <p>Here's the yield curve from the real financial data for {date}:</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Real Yield Curve" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Data source: mk.zero2.csv and mk.maturity.csv</p>
                        
                        <p>The yield curve shows how interest rates vary with different maturities. This data is from actual historical records and can be used for bond pricing and analysis.</p>
                        
                        <p>Would you like to see yield curves from different dates or compare multiple yield curves?</p>
                    </div>
                    """
                else:
                    response = """
                    <div class="result-container">
                        <h3>Error Loading Yield Curve Data</h3>
                        <p>I couldn't load the real yield curve data. Please make sure the data files are available in the correct location.</p>
                    </div>
                    """
            
            elif "treasury" in user_input.lower() or "treasury yield" in user_input.lower():
                # Get treasury yield curve data
                date, maturities, yields = bond_calculator.get_treasury_yield_curve(date_index=0)
                
                if date is not None:
                    # Create a plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(maturities, yields*100, 'r-o', label=f'Treasury Yield Curve on {date}')
                    ax.set_xlabel('Maturity (years)')
                    ax.set_ylabel('Yield (%)')
                    ax.set_title('Treasury Yield Curve')
                    ax.grid(True)
                    ax.legend()
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Treasury Yield Curve Data</h3>
                        <p>Here's the Treasury yield curve from the real financial data for {date}:</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Treasury Yield Curve" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Data source: treasury_yields.txt</p>
                        
                        <p>The Treasury yield curve shows the interest rates on U.S. Treasury securities across different maturities. This is often used as a benchmark for other interest rates in the economy.</p>
                        
                        <p>Would you like to see how this yield curve compares to other dates or how it can be used for bond pricing?</p>
                    </div>
                    """
                else:
                    response = """
                    <div class="result-container">
                        <h3>Error Loading Treasury Yield Data</h3>
                        <p>I couldn't load the Treasury yield curve data. Please make sure the data files are available in the correct location.</p>
                    </div>
                    """
            
            elif "zero coupon" in user_input.lower() or "zero-coupon" in user_input.lower():
                # Get zero coupon bond prices
                maturities, prices = bond_calculator.get_zero_coupon_bond_prices()
                
                if maturities is not None:
                    # Calculate yields
                    maturities_yield, yields = bond_calculator.calculate_yields_from_zero_prices()
                    
                    # Create plots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Price plot
                    ax1.plot(maturities, prices, 'b-o')
                    ax1.set_xlabel('Maturity (years)')
                    ax1.set_ylabel('Price')
                    ax1.set_title('Zero-Coupon Bond Prices')
                    ax1.grid(True)
                    
                    # Yield plot
                    ax2.plot(maturities_yield, yields*100, 'r-o')
                    ax2.set_xlabel('Maturity (years)')
                    ax2.set_ylabel('Yield (%)')
                    ax2.set_title('Zero-Coupon Bond Yields')
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Zero-Coupon Bond Data</h3>
                        <p>Here's the real data for zero-coupon bonds:</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Zero-Coupon Bond Data" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Data source: ZeroPrices.txt</p>
                        
                        <p>The left plot shows the prices of zero-coupon bonds with different maturities. The right plot shows the corresponding yields calculated from these prices.</p>
                        
                        <p>Zero-coupon bonds don't pay periodic interest but instead are sold at a discount to their face value. The yield is implied by the difference between the purchase price and the face value.</p>
                        
                        <p>Would you like to see how these yields compare to the yield curve or how to use this data for bond pricing?</p>
                    </div>
                    """
                else:
                    response = """
                    <div class="result-container">
                        <h3>Error Loading Zero-Coupon Bond Data</h3>
                        <p>I couldn't load the zero-coupon bond data. Please make sure the data files are available in the correct location.</p>
                    </div>
                    """
            
            elif "stock correlation" in user_input.lower() or "gm ford" in user_input.lower():
                # Get stock correlation data
                gm_returns, ford_returns, correlation = bond_calculator.get_stock_bond_correlation('GM', 'Ford')
                
                if gm_returns is not None:
                    # Create a plot
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Returns plot
                    ax1.plot(gm_returns, label='GM Returns')
                    ax1.plot(ford_returns, label='Ford Returns')
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel('Log Returns')
                    ax1.set_title('GM and Ford Log Returns')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Scatter plot
                    ax2.scatter(gm_returns, ford_returns, alpha=0.5)
                    ax2.set_xlabel('GM Log Returns')
                    ax2.set_ylabel('Ford Log Returns')
                    ax2.set_title(f'Correlation: {correlation:.4f}')
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Stock Correlation Analysis</h3>
                        <p>Here's the correlation analysis between GM and Ford stock returns:</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Stock Correlation" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Data source: Stock_Bond.csv</p>
                        
                        <p>The top plot shows the log returns of GM and Ford stocks over time. The bottom plot is a scatter plot of these returns, showing their relationship.</p>
                        
                        <p>The correlation between GM and Ford returns is <strong>{correlation:.4f}</strong>, indicating a {"strong" if correlation > 0.5 else "moderate" if correlation > 0.3 else "weak"} positive correlation. This means that the stocks tend to move {"together" if correlation > 0 else "in opposite directions"}.</p>
                        
                        <p>This correlation analysis is important for portfolio diversification and risk management in investment strategies.</p>
                    </div>
                    """
                else:
                    response = """
                    <div class="result-container">
                        <h3>Error Loading Stock Data</h3>
                        <p>I couldn't load the stock correlation data. Please make sure the data files are available in the correct location.</p>
                    </div>
                    """
            
            elif "random walk" in user_input.lower() or "stock model" in user_input.lower():
                # Get GM stock data for parameters
                gm_returns, _, _ = bond_calculator.get_stock_bond_correlation('GM', 'Ford')
                
                if gm_returns is not None:
                    # Calculate parameters from real data
                    mu = np.mean(gm_returns)
                    sigma = np.std(gm_returns)
                    
                    # Simulate random walks
                    times, paths = bond_calculator.simulate_random_walk(S0=100, mu=mu, sigma=sigma, T=5, num_paths=5)
                    
                    # Create a plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot each path
                    for i in range(paths.shape[0]):
                        ax.plot(times, paths[i], label=f'Path {i+1}')
                    
                    # Plot mean and bounds
                    mean_path = 100 * np.exp(mu * times)
                    upper_bound = 100 * np.exp(mu * times + sigma * np.sqrt(times))
                    lower_bound = 100 * np.exp(mu * times - sigma * np.sqrt(times))
                    
                    ax.plot(times, mean_path, 'k--', linewidth=2, label='Mean Path')
                    ax.plot(times, upper_bound, 'r:', linewidth=2, label='Mean + 1 Std Dev')
                    ax.plot(times, lower_bound, 'r:', linewidth=2, label='Mean - 1 Std Dev')
                    
                    ax.set_xlabel('Time (years)')
                    ax.set_ylabel('Stock Price')
                    ax.set_title('Random Walk Model with Real GM Stock Parameters')
                    ax.legend()
                    ax.grid(True)
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Random Walk Model with Real Data Parameters</h3>
                        <p>Here's a random walk model for stock prices using parameters estimated from real GM stock data:</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Random Walk Model" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Parameters estimated from Stock_Bond.csv</p>
                        
                        <p>The model uses these parameters from real GM stock data:</p>
                        <ul>
                            <li>Drift (μ): {mu:.6f}</li>
                            <li>Volatility (σ): {sigma:.6f}</li>
                        </ul>
                        
                        <p>The plot shows five simulated price paths (colored lines), along with the expected mean path (black dashed line) and one standard deviation bounds (red dotted lines).</p>
                        
                        <p>This geometric Brownian motion model is commonly used in finance to model stock prices and is the foundation for the Black-Scholes option pricing model.</p>
                    </div>
                    """
                else:
                    response = """
                    <div class="result-container">
                        <h3>Error Loading Stock Data for Random Walk</h3>
                        <p>I couldn't load the stock data to estimate parameters. Please make sure the data files are available in the correct location.</p>
                    </div>
                    """
            
            elif "yield to maturity" in user_input.lower():
                response = """
                <p><strong>Yield to Maturity (YTM)</strong> is the total return anticipated on a bond if held until it matures. It's calculated by finding the interest rate that makes the present value of all future cash flows equal to the current price of the bond.</p>
                
                <p>For a bond with semiannual coupon payments, the formula is:</p>
                
                <p class="formula">Price = C/r + (Par - C/r) * (1 + r)^(-2T)</p>
                
                <p>Where:</p>
                <ul>
                    <li>C = semiannual coupon payment</li>
                    <li>r = semiannual yield to maturity</li>
                    <li>T = time to maturity in years</li>
                    <li>Par = par value of the bond</li>
                </ul>
                
                <p>Would you like me to calculate the YTM for a specific bond using real market data?</p>
                """
            elif "calculate" in user_input.lower() and "price" in user_input.lower() and "bond" in user_input.lower():
                # Extract parameters from the query (this is a simplified example)
                coupon_rate = 0.05  # Default values
                ytm = 0.04
                T = 10  # Default maturity
                
                # Try to extract values from the query
                if "coupon" in user_input.lower() and "%" in user_input:
                    try:
                        coupon_text = user_input.split("coupon")[1].split("%")[0]
                        coupon_rate = float(coupon_text.strip()) / 100
                    except:
                        pass
                
                if "ytm" in user_input.lower() and "%" in user_input:
                    try:
                        ytm_text = user_input.split("ytm")[1].split("%")[0]
                        ytm = float(ytm_text.strip()) / 100
                    except:
                        pass
                
                if "years" in user_input.lower() or "maturity" in user_input.lower():
                    try:
                        # Look for numbers followed by "year" or "years"
                        import re
                        maturity_matches = re.findall(r'(\d+)\s*(?:year|years)', user_input.lower())
                        if maturity_matches:
                            T = int(maturity_matches[0])
                    except:
                        pass
                
                # Calculate bond price
                par = 1000
                c = coupon_rate * par / 2  # Semiannual coupon
                r = ytm / 2  # Semiannual yield
                
                price = bond_calculator.bondvalue(c, T, r, par)
                
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 6))
                r_values = np.linspace(max(0.001, r-0.02), r+0.02, 100)
                prices = [bond_calculator.bondvalue(c, T, r_val, par) for r_val in r_values]
                
                ax.plot(r_values*2*100, prices, 'b-')  # Convert to annual percentage
                ax.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
                ax.axvline(x=ytm*100, color='g', linestyle='--', label=f'YTM = {ytm*100:.2f}%')
                ax.set_xlabel('Yield to Maturity (%)')
                ax.set_ylabel('Bond Price ($)')
                ax.set_title('Bond Price vs. Yield to Maturity')
                ax.grid(True)
                ax.legend()
                
                # Calculate duration
                duration = bond_calculator.calculate_bond_duration(c, T, r, par)
                
                # Convert plot to base64 for embedding
                img_str = bond_calculator.get_plot_as_base64(fig)
                
                response = f"""
                <div class="result-container">
                    <h3>Bond Price Calculation</h3>
                    <p><strong>Parameters:</strong></p>
                    <ul>
                        <li>Par Value: ${par}</li>
                        <li>Coupon Rate: {coupon_rate*100:.2f}% (${coupon_rate*par:.2f} annually, ${c:.2f} semiannually)</li>
                        <li>Yield to Maturity: {ytm*100:.2f}%</li>
                        <li>Time to Maturity: {T} years</li>
                    </ul>
                    
                    <p><strong>Calculated Bond Price:</strong> ${price:.2f}</p>
                    <p><strong>Bond Duration:</strong> {duration:.2f} years</p>
                    
                    <p>The bond is selling at a {"premium" if price > par else "discount"} because the coupon rate is {"higher" if coupon_rate > ytm else "lower"} than the yield to maturity.</p>
                    
                    <img src="data:image/png;base64,{img_str}" alt="Bond Price vs YTM" style="width:100%; max-width:800px;">
                </div>
                """
            elif "calculate" in user_input.lower() and "ytm" in user_input.lower() and "bond" in user_input.lower():
                # Extract parameters from the query
                price = 950  # Default values
                coupon_rate = 0.05
                T = 10
                
                # Try to extract values from the query
                if "price" in user_input.lower() and "$" in user_input:
                    try:
                        price_text = user_input.split("price")[1].split("$")[1].split()[0]
                        price = float(price_text.strip())
                    except:
                        pass
                
                if "coupon" in user_input.lower() and "%" in user_input:
                    try:
                        coupon_text = user_input.split("coupon")[1].split("%")[0]
                        coupon_rate = float(coupon_text.strip()) / 100
                    except:
                        pass
                
                if "years" in user_input.lower() or "maturity" in user_input.lower():
                    try:
                        import re
                        maturity_matches = re.findall(r'(\d+)\s*(?:year|years)', user_input.lower())
                        if maturity_matches:
                            T = int(maturity_matches[0])
                    except:
                        pass
                
                # Calculate YTM
                par = 1000
                c = coupon_rate * par / 2  # Semiannual coupon
                
                ytm = bond_calculator.find_ytm_using_root(price, c, T, par)
                annual_ytm = ytm * 2
                
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ytm_values = np.linspace(max(0.001, ytm-0.02), ytm+0.02, 100)
                prices = [bond_calculator.bondvalue(c, T, r, par) for r in ytm_values]
                
                ax.plot(ytm_values*2*100, prices, 'b-')  # Convert to annual percentage
                ax.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
                ax.axvline(x=annual_ytm*100, color='g', linestyle='--', label=f'YTM = {annual_ytm*100:.2f}%')
                ax.set_xlabel('Yield to Maturity (%)')
                ax.set_ylabel('Bond Price ($)')
                ax.set_title('Bond Price vs. Yield to Maturity')
                ax.grid(True)
                ax.legend()
                
                # Calculate duration
                duration = bond_calculator.calculate_bond_duration(c, T, ytm, par)
                
                # Calculate current yield
                current_yield = (coupon_rate * par) / price
                
                # Convert plot to base64 for embedding
                img_str = bond_calculator.get_plot_as_base64(fig)
                
                response = f"""
                <div class="result-container">
                    <h3>Yield to Maturity Calculation</h3>
                    <p><strong>Parameters:</strong></p>
                    <ul>
                        <li>Par Value: ${par}</li>
                        <li>Bond Price: ${price:.2f}</li>
                        <li>Coupon Rate: {coupon_rate*100:.2f}% (${coupon_rate*par:.2f} annually, ${c:.2f} semiannually)</li>
                        <li>Time to Maturity: {T} years</li>
                    </ul>
                    
                    <p><strong>Calculated Yield to Maturity:</strong> {annual_ytm*100:.4f}%</p>
                    <p><strong>Current Yield:</strong> {current_yield*100:.4f}%</p>
                    <p><strong>Bond Duration:</strong> {duration:.2f} years</p>
                    
                    <p>The bond is selling at a {"premium" if price > par else "discount"} because the coupon rate is {"higher" if coupon_rate > annual_ytm else "lower"} than the yield to maturity.</p>
                    
                    <img src="data:image/png;base64,{img_str}" alt="Bond Price vs YTM" style="width:100%; max-width:800px;">
                </div>
                """
            elif "forward rate" in user_input.lower():
                # Get real yield curve data
                date, maturities, yields = bond_calculator.get_real_yield_curve_data(date_index=5)
                
                if date is not None:
                    # Calculate forward rates
                    forward_rates = bond_calculator.calculate_forward_rates(yields, maturities)
                    forward_maturities = [(maturities[i] + maturities[i-1])/2 for i in range(1, len(maturities))]
                    
                    # Create a plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(maturities, yields, 'b-o', label='Spot Rates (Yield Curve)')
                    ax.plot(forward_maturities, forward_rates, 'r-o', label='Forward Rates')
                    ax.set_xlabel('Maturity (years)')
                    ax.set_ylabel('Rate')
                    ax.set_title(f'Yield Curve and Forward Rates on {date}')
                    ax.grid(True)
                    ax.legend()
                    
                    # Convert plot to base64 for embedding
                    img_str = bond_calculator.get_plot_as_base64(fig)
                    
                    response = f"""
                    <div class="result-container">
                        <h3>Forward Rates from Real Data</h3>
                        <p><strong>Forward Rate</strong> represents the interest rate for a future period that is implied by the current term structure of interest rates.</p>
                        
                        <img src="data:image/png;base64,{img_str}" alt="Forward Rates" style="width:100%; max-width:800px;">
                        
                        <p class="data-source">Data source: mk.zero2.csv and mk.maturity.csv</p>
                        
                        <p>The forward rate between times t₁ and t₂ is calculated from spot rates using:</p>
                        <p class="formula">Forward Rate(t₁,t₂) = (Spot Rate(t₂) × t₂ - Spot Rate(t₁) × t₁) / (t₂ - t₁)</p>
                        
                        <p>The plot shows both the spot rates (yield curve) and the derived forward rates from real market data on {date}. Forward rates represent market expectations of future interest rates.</p>
                    </div>
                    """
                else:
                    response = """
                    <p><strong>Forward Rate</strong> represents the interest rate for a future period that is implied by the current term structure of interest rates.</p>
                    
                    <p>The forward rate function r(t) gives the instantaneous forward rate at time t. Common forms include:</p>
                    <ul>
                        <li>Linear: r(t) = a + bt</li>
                        <li>Quadratic: r(t) = a + bt + ct²</li>
                        <li>Piecewise: r(t) = a + bt - c(t-T)₊</li>
                    </ul>
                    
                    <p>The spot rate (or yield to maturity) for a zero-coupon bond maturing at time T is the average of the forward rates from 0 to T:</p>
                    
                    <p class="formula">R(T) = (1/T) ∫₀ᵀ r(t) dt</p>
                    
                    <p>Would you like me to calculate spot rates or bond prices using a specific forward rate function?</p>
                    """
            elif "duration" in user_input.lower():
                response = """
                <p><strong>Bond Duration</strong> is a measure of the sensitivity of a bond's price to changes in interest rates. It represents the weighted average time to receive the bond's cash flows.</p>
                
                <p>For a bond with cash flows C₁, C₂, ..., Cₙ at times T₁, T₂, ..., Tₙ, the duration is defined as:</p>
                
                <p class="formula">DUR = ∑ᵢ₌₁ⁿ ωᵢTᵢ</p>
                
                <p>where ωᵢ = NPVᵢ / ∑ⱼ₌₁ⁿ NPVⱼ and NPVᵢ = Cᵢ exp(-Tᵢ yTᵢ)</p>
                
                <p>Duration is related to the sensitivity of bond price to changes in yield through the formula:</p>
                
                <p class="formula">d/dδ [∑ᵢ₌₁ⁿ Cᵢ exp(-Tᵢ(yTᵢ + δ))]|δ=0 = -DUR ∑ᵢ₌₁ⁿ Cᵢ exp{-TᵢyTᵢ}</p>
                
                <p>Would you like me to calculate the duration for a specific bond?</p>
                """
            elif "data" in user_input.lower() or "available data" in user_input.lower():
                response = """
                <div class="result-container">
                    <h3>Available Financial Data</h3>
                    <p>The bond calculator has access to the following real financial datasets:</p>
                    
                    <h4>Yield Curve Data:</h4>
                    <ul>
                        <li><strong>mk.maturity.csv</strong> - Maturity values for yield curves</li>
                        <li><strong>mk.zero2.csv</strong> - Historical yield curves at various dates</li>
                        <li><strong>treasury_yields.txt</strong> - U.S. Treasury yield data</li>
                        <li><strong>yields.txt</strong> - Additional yield data</li>
                    </ul>
                    
                    <h4>Bond Price Data:</h4>
                    <ul>
                        <li><strong>bondprices.txt</strong> - Prices of various bonds</li>
                        <li><strong>ZeroPrices.txt</strong> - Prices of zero-coupon bonds</li>
                    </ul>
                    
                    <h4>Stock and Market Data:</h4>
                    <ul>
                        <li><strong>Stock_Bond.csv</strong> - Historical stock and bond data</li>
                        <li><strong>Stock_Bond_2004_to_2006.csv</strong> - Stock and bond data for 2004-2006</li>
                    </ul>
                    
                    <p>You can ask questions about any of these datasets, such as:</p>
                    <ul>
                        <li>"Show me real yield curve data"</li>
                        <li>"Display treasury yield curves"</li>
                        <li>"Show zero-coupon bond prices"</li>
                        <li>"Calculate correlation between GM and Ford stocks"</li>
                        <li>"Simulate a random walk with real stock parameters"</li>
                    </ul>
                </div>
                """
            else:
                response = """
                <p>I can help with various bond calculations using real financial data, including:</p>
                <ul>
                    <li>Displaying real yield curves from historical data</li>
                    <li>Analyzing Treasury yield data</li>
                    <li>Examining zero-coupon bond prices and yields</li>
                    <li>Calculating stock correlations from market data</li>
                    <li>Simulating random walks with parameters from real stocks</li>
                    <li>Calculating bond prices and yields to maturity</li>
                    <li>Computing forward rates from spot rates</li>
                    <li>Determining bond duration and other metrics</li>
                </ul>
                
                <p>Try asking questions like:</p>
                <ul>
                    <li>"Show me real yield curve data"</li>
                    <li>"Display treasury yield curves"</li>
                    <li>"Show zero-coupon bond prices"</li>
                    <li>"Calculate correlation between GM and Ford stocks"</li>
                    <li>"Simulate a random walk with real stock parameters"</li>
                    <li>"What is yield to maturity?"</li>
                    <li>"Calculate the price of a bond with coupon rate 5% and YTM 4%"</li>
                </ul>
                
                <p>You can also select a specific category from the sidebar to explore different types of bond calculations.</p>
                """
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the display
            st.experimental_rerun()

elif category == "Real Data Analysis":
    st.markdown("## Real Financial Data Analysis")
    
    analysis_type = st.sidebar.selectbox(
        "Select analysis type:",
        ["Yield Curve Analysis", "Zero-Coupon Bond Analysis", "Treasury Yield Analysis", 
         "Stock Correlation Analysis", "Random Walk Simulation"]
    )
    
    if analysis_type == "Yield Curve Analysis":
        st.markdown("### Yield Curve Analysis from Real Data")
        
        # Date selection
        date_index = st.slider("Select date index", 0, 10, 5)
        
        if st.button("Generate Yield Curve"):
            # Get real yield curve data
            date, maturities, yields = bond_calculator.get_real_yield_curve_data(date_index=date_index)
            
            if date is not None:
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(maturities, yields, 'b-o', label=f'Yield Curve on {date}')
                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('Yield')
                ax.set_title('Real Yield Curve Data')
                ax.grid(True)
                ax.legend()
                
                # Display the plot
                st.pyplot(fig)
                
                # Calculate forward rates
                forward_rates = bond_calculator.calculate_forward_rates(yields, maturities)
                forward_maturities = [(maturities[i] + maturities[i-1])/2 for i in range(1, len(maturities))]
                
                # Create a plot for forward rates
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(maturities, yields, 'b-o', label='Spot Rates (Yield Curve)')
                ax2.plot(forward_maturities, forward_rates, 'r-o', label='Forward Rates')
                ax2.set_xlabel('Maturity (years)')
                ax2.set_ylabel('Rate')
                ax2.set_title(f'Yield Curve and Forward Rates on {date}')
                ax2.grid(True)
                ax2.legend()
                
                # Display the plot
                st.pyplot(fig2)
                
                # Display data source
                st.markdown('<p class="data-source">Data source: mk.zero2.csv and mk.maturity.csv</p>', unsafe_allow_html=True)
            else:
                st.error("Could not load yield curve data. Please check data files.")
    
    elif analysis_type == "Zero-Coupon Bond Analysis":
        st.markdown("### Zero-Coupon Bond Analysis from Real Data")
        
        if st.button("Analyze Zero-Coupon Bonds"):
            # Get zero coupon bond prices
            maturities, prices = bond_calculator.get_zero_coupon_bond_prices()
            
            if maturities is not None:
                # Calculate yields
                maturities_yield, yields = bond_calculator.calculate_yields_from_zero_prices()
                
                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Price plot
                ax1.plot(maturities, prices, 'b-o')
                ax1.set_xlabel('Maturity (years)')
                ax1.set_ylabel('Price')
                ax1.set_title('Zero-Coupon Bond Prices')
                ax1.grid(True)
                
                # Yield plot
                ax2.plot(maturities_yield, yields*100, 'r-o')
                ax2.set_xlabel('Maturity (years)')
                ax2.set_ylabel('Yield (%)')
                ax2.set_title('Zero-Coupon Bond Yields')
                ax2.grid(True)
                
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                
                # Display data in a table
                data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Price': prices,
                    'Yield (%)': yields*100
                })
                
                st.markdown("### Zero-Coupon Bond Data")
                st.dataframe(data)
                
                # Display data source
                st.markdown('<p class="data-source">Data source: ZeroPrices.txt</p>', unsafe_allow_html=True)
            else:
                st.error("Could not load zero-coupon bond data. Please check data files.")
    
    elif analysis_type == "Treasury Yield Analysis":
        st.markdown("### Treasury Yield Analysis from Real Data")
        
        # Date selection
        date_index = st.slider("Select date index", 0, 9, 0)
        
        if st.button("Generate Treasury Yield Curve"):
            # Get treasury yield curve data
            date, maturities, yields = bond_calculator.get_treasury_yield_curve(date_index=date_index)
            
            if date is not None:
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(maturities, yields*100, 'r-o', label=f'Treasury Yield Curve on {date}')
                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('Yield (%)')
                ax.set_title('Treasury Yield Curve')
                ax.grid(True)
                ax.legend()
                
                # Display the plot
                st.pyplot(fig)
                
                # Display data in a table
                data = pd.DataFrame({
                    'Maturity (years)': maturities,
                    'Yield (%)': yields*100
                })
                
                st.markdown("### Treasury Yield Data")
                st.dataframe(data)
                
                # Display data source
                st.markdown('<p class="data-source">Data source: treasury_yields.txt</p>', unsafe_allow_html=True)
            else:
                st.error("Could not load Treasury yield data. Please check data files.")
    
    elif analysis_type == "Stock Correlation Analysis":
        st.markdown("### Stock Correlation Analysis from Real Data")
        
        # Stock selection
        stock1 = st.selectbox("Select first stock", ["GM", "Ford", "GE", "IBM"], index=0)
        stock2 = st.selectbox("Select second stock", ["GM", "Ford", "GE", "IBM"], index=1)
        
        if st.button("Analyze Stock Correlation"):
            # Get stock correlation data
            stock1_returns, stock2_returns, correlation = bond_calculator.get_stock_bond_correlation(stock1, stock2)
            
            if stock1_returns is not None:
                # Create a plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Returns plot
                ax1.plot(stock1_returns, label=f'{stock1} Returns')
                ax1.plot(stock2_returns, label=f'{stock2} Returns')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Log Returns')
                ax1.set_title(f'{stock1} and {stock2} Log Returns')
                ax1.legend()
                ax1.grid(True)
                
                # Scatter plot
                ax2.scatter(stock1_returns, stock2_returns, alpha=0.5)
                ax2.set_xlabel(f'{stock1} Log Returns')
                ax2.set_ylabel(f'{stock2} Log Returns')
                ax2.set_title(f'Correlation: {correlation:.4f}')
                ax2.grid(True)
                
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                
                # Display correlation statistics
                st.markdown(f"### Correlation Analysis: {stock1} vs {stock2}")
                st.markdown(f"**Correlation Coefficient:** {correlation:.4f}")
                
                if correlation > 0.7:
                    st.markdown("**Interpretation:** Strong positive correlation")
                elif correlation > 0.3:
                    st.markdown("**Interpretation:** Moderate positive correlation")
                elif correlation > 0:
                    st.markdown("**Interpretation:** Weak positive correlation")
                elif correlation > -0.3:
                    st.markdown("**Interpretation:** Weak negative correlation")
                elif correlation > -0.7:
                    st.markdown("**Interpretation:** Moderate negative correlation")
                else:
                    st.markdown("**Interpretation:** Strong negative correlation")
                
                # Display data source
                st.markdown('<p class="data-source">Data source: Stock_Bond.csv</p>', unsafe_allow_html=True)
            else:
                st.error(f"Could not load stock data for {stock1} and {stock2}. Please check data files.")
    
    elif analysis_type == "Random Walk Simulation":
        st.markdown("### Random Walk Simulation with Real Stock Parameters")
        
        # Stock selection
        stock = st.selectbox("Select stock for parameter estimation", ["GM", "Ford", "GE", "IBM"], index=0)
        
        # Simulation parameters
        initial_price = st.number_input("Initial price", min_value=1.0, max_value=1000.0, value=100.0)
        time_horizon = st.slider("Time horizon (years)", min_value=1, max_value=10, value=5)
        num_paths = st.slider("Number of simulation paths", min_value=1, max_value=10, value=5)
        
        if st.button("Run Simulation"):
            # Get stock data for parameters
            stock_returns, _, _ = bond_calculator.get_stock_bond_correlation(stock, 'Ford')
            
            if stock_returns is not None:
                # Calculate parameters from real data
                mu = np.mean(stock_returns)
                sigma = np.std(stock_returns)
                
                # Simulate random walks
                times, paths = bond_calculator.simulate_random_walk(
                    S0=initial_price, 
                    mu=mu, 
                    sigma=sigma, 
                    T=time_horizon, 
                    num_paths=num_paths
                )
                
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot each path
                for i in range(paths.shape[0]):
                    ax.plot(times, paths[i], label=f'Path {i+1}')
                
                # Plot mean and bounds
                mean_path = initial_price * np.exp(mu * times)
                upper_bound = initial_price * np.exp(mu * times + sigma * np.sqrt(times))
                lower_bound = initial_price * np.exp(mu * times - sigma * np.sqrt(times))
                
                ax.plot(times, mean_path, 'k--', linewidth=2, label='Mean Path')
                ax.plot(times, upper_bound, 'r:', linewidth=2, label='Mean + 1 Std Dev')
                ax.plot(times, lower_bound, 'r:', linewidth=2, label='Mean - 1 Std Dev')
                
                ax.set_xlabel('Time (years)')
                ax.set_ylabel('Stock Price')
                ax.set_title(f'Random Walk Model with Real {stock} Stock Parameters')
                ax.legend()
                ax.grid(True)
                
                # Display the plot
                st.pyplot(fig)
                
                # Display parameter information
                st.markdown(f"### Random Walk Parameters for {stock}")
                st.markdown(f"**Drift (μ):** {mu:.6f}")
                st.markdown(f"**Volatility (σ):** {sigma:.6f}")
                
                # Display data source
                st.markdown('<p class="data-source">Parameters estimated from Stock_Bond.csv</p>', unsafe_allow_html=True)
            else:
                st.error(f"Could not load stock data for {stock}. Please check data files.")

elif category == "Bond Yield & Pricing (Q1-5)":
    st.markdown("## Bond Yield & Pricing Calculations (Questions 1-5)")
    
    question = st.sidebar.selectbox(
        "Select a question:",
        ["Question 1: Forward Rate and Bond Pricing",
         "Question 2: Forward Rate, Yield Curve, and Returns",
         "Question 3: Coupon Bond Analysis",
         "Question 4: Forward Rate and Spot Rate",
         "Question 5: Bond Pricing with Spot Rates"]
    )
    
    if "Question 1" in question:
        st.markdown("""
        <div class="question-card">
            <h3>Question 1: Forward Rate and Bond Pricing</h3>
            <p>Given forward rate r(t) = 0.028 + 0.00042t:</p>
            <p>(a) What is the yield to maturity of a bond maturing in 20 years?</p>
            <p>(b) What is the price of a par $1,000 zero-coupon bond maturing in 15 years?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Solve Question 1"):
            # Define forward rate function
            def forward_rate(t):
                return 0.028 + 0.00042 * t
            
            # (a) Calculate the yield to maturity of a bond maturing in 20 years
            ytm_20yr = bond_calculator.calculate_spot_rate(forward_rate, 20)
            
            # (b) Calculate the price of a par $1,000 zero-coupon bond maturing in 15 years
            spot_rate_15yr = bond_calculator.calculate_spot_rate(forward_rate, 15)
            price_15yr = 1000 * np.exp(-spot_rate_15yr * 15)
            
            # Create a plot to visualize the forward rate and spot rates
            fig, ax = plt.subplots(figsize=(10, 6))
            t_values = np.linspace(0, 25, 100)
            forward_rates = [forward_rate(t) for t in t_values]
            spot_rates = [bond_calculator.calculate_spot_rate(forward_rate, t) if t > 0 else forward_rate(0) for t in t_values]
            
            ax.plot(t_values, forward_rates, 'b-', label='Forward Rate: r(t) = 0.028 + 0.00042t')
            ax.plot(t_values, spot_rates, 'r--', label='Spot Rate (YTM)')
            ax.axvline(x=20, color='g', linestyle=':', label='20-year maturity')
            ax.axhline(y=ytm_20yr, color='g', linestyle=':', label=f'20-year spot rate: {ytm_20yr:.4f}')
            ax.axvline(x=15, color='m', linestyle=':', label='15-year maturity')
            ax.axhline(y=spot_rate_15yr, color='m', linestyle=':', label=f'15-year spot rate: {spot_rate_15yr:.4f}')
            ax.set_xlabel('Maturity (years)')
            ax.set_ylabel('Rate')
            ax.set_title('Forward Rate and Spot Rates')
            ax.grid(True)
            ax.legend()
            
            # Display the plot
            st.pyplot(fig)
            
            # Display results
            st.markdown(f"""
            <div class="result-container">
                <h3>Solution to Question 1</h3>
                
                <h4>(a) Yield to maturity of a bond maturing in 20 years:</h4>
                <p>{ytm_20yr:.6f} or {ytm_20yr*100:.4f}%</p>
                
                <h4>(b) Price of a par $1,000 zero-coupon bond maturing in 15 years:</h4>
                <p>${price_15yr:.2f}</p>
                
                <h4>Explanation:</h4>
                <p>The yield to maturity (spot rate) is calculated as the average of the forward rates from time 0 to maturity T:</p>
                <p class="formula">R(T) = (1/T) ∫₀ᵀ r(t) dt</p>
                
                <p>For a 20-year bond with r(t) = 0.028 + 0.00042t:</p>
                <p class="formula">R(20) = (1/20) ∫₀²⁰ (0.028 + 0.00042t) dt = (1/20) [0.028t + 0.00042t²/2]₀²⁰ = {ytm_20yr:.6f}</p>
                
                <p>For a zero-coupon bond, the price is calculated as:</p>
                <p class="formula">Price = Par × e^(-R(T) × T)</p>
                
                <p>For a 15-year zero-coupon bond with par value $1,000:</p>
                <p class="formula">Price = $1,000 × e^(-{spot_rate_15yr:.6f} × 15) = ${price_15yr:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add implementations for other questions in this category

elif category == "Yield to Maturity (Q6-8)":
    st.markdown("## Yield to Maturity and Bond Pricing (Questions 6-8)")
    
    question = st.sidebar.selectbox(
        "Select a question:",
        ["Question 6: Finding Coupon Payment Given YTM",
         "Question 7: YTM of a 30-year Bond",
         "Question 8: YTM of a 20-year Bond"]
    )
    
    # Add implementations for questions in this category

elif category == "Bond Calculations (Q9-13)":
    st.markdown("## More Bond Calculations (Questions 9-13)")
    
    question = st.sidebar.selectbox(
        "Select a question:",
        ["Question 9: Bond Pricing with a Given Yield",
         "Question 10: Yield to Maturity and Current Yield",
         "Question 11: Zero-Coupon Bond Pricing with Forward Rate",
         "Question 12: Bond Return with Changing Forward Rate",
         "Question 13: Yield to Maturity with Piecewise Forward Rate"]
    )
    
    # Add implementations for questions in this category
    
elif category == "Advanced Analysis (Q14-20)":
    st.markdown("## Advanced Bond Analysis (Questions 14-20)")
    
    question = st.sidebar.selectbox(
        "Select a question:",
        ["Question 14: Zero-Coupon Bond Investment Analysis",
         "Question 15: Bond Duration Derivative",
         "Question 16: Zero-Coupon Bond with Yield Curve",
         "Question 17: Coupon Bond Analysis",
         "Question 18: Forward Rate and Spot Rate",
         "Question 19: Bond Pricing with Spot Rates",
         "Question 20: Calculate Spot Rates from Bond Prices"]
    )
    
    # Add implementations for questions in this category

# Add a footer
st.markdown("""
---
<p style="text-align: center; color: #666;">Bond Calculator Chat UI | Created for educational purposes</p>
""", unsafe_allow_html=True)
