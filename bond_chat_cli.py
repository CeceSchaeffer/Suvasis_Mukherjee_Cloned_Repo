#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq
import os
import pandas as pd
import sys
import re
from pathlib import Path
import shutil

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
            # Create a list of files to try loading
            files_to_load = {
                'mk_maturity': 'mk.maturity.csv',
                'mk_zero2': 'mk.zero2.csv',
                'bond_prices': 'bondprices.txt',
                'zero_prices': 'ZeroPrices.txt',
                'yields': 'yields.txt',
                'treasury_yields': 'treasury_yields.txt',
                'stock_bond': 'Stock_Bond.csv'
            }
            
            # Try to load each file
            for key, filename in files_to_load.items():
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    if filename.endswith('.txt'):
                        self.datasets[key] = pd.read_csv(filepath, sep='\s+')
                    else:
                        self.datasets[key] = pd.read_csv(filepath, header=0)
            
            # Check if we loaded at least some of the files
            self.data_loaded = len(self.datasets) > 0
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
        # Check for division by zero
        if isinstance(r, (int, float)) and r == 0:
            # For zero yield, it's just the sum of coupons plus par
            return c * (2 * T) + par
        
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
        if not self.data_loaded or 'mk_zero2' not in self.datasets or 'mk_maturity' not in self.datasets:
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
        if not self.data_loaded or 'treasury_yields' not in self.datasets:
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
        if not self.data_loaded or 'zero_prices' not in self.datasets:
            return None, None
        
        maturities = self.datasets['zero_prices']['maturity'].values
        prices = self.datasets['zero_prices']['price'].values
        
        return maturities, prices
    
    def get_bond_prices(self):
        """Get bond prices from the dataset"""
        if not self.data_loaded or 'bond_prices' not in self.datasets:
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
    
    def list_available_data(self):
        """List all available datasets loaded"""
        if not self.data_loaded:
            return "No data loaded."
        
        return list(self.datasets.keys())
    
    def add_dataset(self, name, filepath):
        """Add a new dataset from a file"""
        try:
            if filepath.endswith('.txt'):
                self.datasets[name] = pd.read_csv(filepath, sep='\s+')
            else:
                self.datasets[name] = pd.read_csv(filepath, header=0)
            return True
        except Exception as e:
            print(f"Error loading dataset {name} from {filepath}: {e}")
            return False

class ChatInterface:
    """A simple chat interface for interacting with the bond calculator"""
    
    def __init__(self):
        self.calculator = BondCalculator()
        self.chat_history = []
        self.last_figure = None
        self.commands = {
            'calculate bond': self.calculate_bond_price,
            'find ytm': self.find_ytm,
            'calculate duration': self.calculate_duration,
            'show yield curve': self.show_yield_curve,
            'show zero coupon': self.show_zero_coupon,
            'show treasury': self.show_treasury,
            'plot price vs yield': self.plot_price_vs_yield,
            'calculate forward rates': self.calculate_forward_rates,
            'calculate correlation': self.calculate_correlation,
            'simulate random walk': self.simulate_random_walk,
            'list data': self.list_data,
            'help': self.show_help,
            'upload': self.upload_data,
            'exit': self.exit_chat,
            'quit': self.exit_chat
        }
    
    def add_to_history(self, sender, message):
        """Add a message to the chat history"""
        self.chat_history.append({'sender': sender, 'message': message})
    
    def display_chat(self):
        """Display the chat history"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n" + "="*80)
        print(" "*30 + "BOND CALCULATOR CHAT" + " "*30)
        print("="*80 + "\n")
        
        for entry in self.chat_history[-15:]:  # Show last 15 messages
            prefix = "🤖 Bot: " if entry['sender'] == 'bot' else "👤 You: "
            print(f"{prefix}{entry['message']}")
        
        print("\n" + "-"*80)
    
    def start(self):
        """Start the chat interface"""
        self.add_to_history('bot', "Welcome to the Bond Calculator Chat! Type 'help' to see available commands.")
        
        while True:
            self.display_chat()
            user_input = input("Enter your message (or 'exit' to quit): ")
            
            if not user_input.strip():
                continue
                
            self.add_to_history('user', user_input)
            
            # Process the user input
            self.process_input(user_input)
    
    def process_input(self, user_input):
        """Process the user input and execute the appropriate command"""
        user_input = user_input.lower().strip()
        
        # Check if the input matches any command
        command_found = False
        for cmd, func in self.commands.items():
            if user_input.startswith(cmd):
                func(user_input)
                command_found = True
                break
        
        if not command_found:
            self.handle_general_query(user_input)
    
    def handle_general_query(self, query):
        """Handle general queries that don't match specific commands"""
        if 'yield' in query and 'maturity' in query:
            self.add_to_history('bot', 
                "Yield to Maturity (YTM) is the total return anticipated on a bond if held until maturity. "
                "It's calculated by finding the interest rate that makes the present value of all future cash flows "
                "equal to the current price of the bond.\n\n"
                "Type 'find ytm' followed by bond details to calculate it.")
        
        elif 'duration' in query:
            self.add_to_history('bot', 
                "Bond duration measures the sensitivity of a bond's price to interest rate changes. "
                "It represents the weighted average time to receive the bond's cash flows.\n\n"
                "Type 'calculate duration' followed by bond details to calculate it.")
        
        elif 'forward rate' in query:
            self.add_to_history('bot', 
                "Forward rates represent future interest rates implied by the current term structure. "
                "They can be derived from spot rates (yields).\n\n"
                "Type 'calculate forward rates' to see forward rates calculated from real data.")
        
        elif 'data' in query:
            self.list_data(query)
        
        else:
            self.add_to_history('bot', 
                "I'm not sure how to respond to that. Type 'help' to see available commands.")
    
    def show_help(self, _):
        """Show available commands"""
        help_text = """
Available commands:
- calculate bond [coupon rate] [maturity] [ytm] [par]
- find ytm [price] [coupon rate] [maturity] [par]
- calculate duration [coupon rate] [maturity] [ytm] [par]
- show yield curve
- show zero coupon
- show treasury
- plot price vs yield [coupon rate] [maturity] [par]
- calculate forward rates
- calculate correlation [stock1] [stock2]
- simulate random walk [stock] [initial price] [time horizon] [num paths]
- list data
- upload [dataset name] [file path]
- help
- exit/quit

Examples:
- calculate bond coupon 5% maturity 10 ytm 4%
- find ytm price 950 coupon 5% maturity 10
- calculate duration coupon 5% maturity 10 ytm 4%
        """
        self.add_to_history('bot', help_text)
    
    def extract_param_value(self, text, param, is_percentage=False, default=None):
        """Extract parameter value from text"""
        pattern = f"{param}\\s+([\\d.]+)%?" if not is_percentage else f"{param}\\s+([\\d.]+)%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if is_percentage:
                value /= 100  # Convert percentage to decimal
            return value
        return default
    
    def calculate_bond_price(self, query):
        """Calculate bond price from user input"""
        coupon_rate = self.extract_param_value(query, "coupon", is_percentage=True, default=0.05)
        maturity = self.extract_param_value(query, "maturity", default=10)
        ytm = self.extract_param_value(query, "ytm", is_percentage=True, default=0.04)
        par = self.extract_param_value(query, "par", default=1000)
        
        # Convert to semi-annual
        semiannual_coupon = coupon_rate * par / 2
        semiannual_ytm = ytm / 2
        
        price = self.calculator.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
        
        response = f"""
Bond Price Calculation Results:
------------------------------
Parameters:
- Par Value: ${par:.2f}
- Annual Coupon Rate: {coupon_rate*100:.2f}%
- Time to Maturity: {maturity:.2f} years
- Annual Yield to Maturity: {ytm*100:.2f}%

Result:
- Bond Price: ${price:.2f}
- The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.
        """
        
        self.add_to_history('bot', response)
        
        # Generate and save a simple price vs yield curve
        self.generate_price_vs_yield_plot(coupon_rate, maturity, ytm, par)
    
    def find_ytm(self, query):
        """Find yield to maturity from user input"""
        price = self.extract_param_value(query, "price", default=950)
        coupon_rate = self.extract_param_value(query, "coupon", is_percentage=True, default=0.05)
        maturity = self.extract_param_value(query, "maturity", default=10)
        par = self.extract_param_value(query, "par", default=1000)
        
        # Convert to semi-annual
        semiannual_coupon = coupon_rate * par / 2
        
        # Calculate YTM
        semiannual_ytm = self.calculator.find_ytm_using_root(price, semiannual_coupon, maturity, par)
        annual_ytm = semiannual_ytm * 2
        
        # Calculate current yield
        current_yield = (coupon_rate * par) / price
        
        response = f"""
Yield to Maturity Calculation Results:
------------------------------------
Parameters:
- Bond Price: ${price:.2f}
- Par Value: ${par:.2f}
- Annual Coupon Rate: {coupon_rate*100:.2f}%
- Time to Maturity: {maturity:.2f} years

Results:
- Yield to Maturity: {annual_ytm*100:.4f}%
- Current Yield: {current_yield*100:.4f}%
- The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.
        """
        
        self.add_to_history('bot', response)
        
        # Generate and save a simple price vs yield curve highlighting the YTM
        self.generate_price_vs_yield_plot(coupon_rate, maturity, annual_ytm, par, price)
    
    def calculate_duration(self, query):
        """Calculate bond duration from user input"""
        coupon_rate = self.extract_param_value(query, "coupon", is_percentage=True, default=0.05)
        maturity = self.extract_param_value(query, "maturity", default=10)
        ytm = self.extract_param_value(query, "ytm", is_percentage=True, default=0.04)
        par = self.extract_param_value(query, "par", default=1000)
        
        # Convert to semi-annual
        semiannual_coupon = coupon_rate * par / 2
        semiannual_ytm = ytm / 2
        
        # Calculate duration
        duration = self.calculator.calculate_bond_duration(semiannual_coupon, maturity, semiannual_ytm, par)
        
        # Calculate modified duration
        modified_duration = duration / (1 + semiannual_ytm)
        
        # Calculate price
        price = self.calculator.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
        
        response = f"""
Bond Duration Calculation Results:
--------------------------------
Parameters:
- Par Value: ${par:.2f}
- Annual Coupon Rate: {coupon_rate*100:.2f}%
- Time to Maturity: {maturity:.2f} years
- Annual Yield to Maturity: {ytm*100:.2f}%

Results:
- Bond Price: ${price:.2f}
- Macaulay Duration: {duration:.4f} years
- Modified Duration: {modified_duration:.4f}
- Price Sensitivity: For a 1% increase in yield, the bond price would decrease by approximately {modified_duration*100:.2f}%

Duration represents the weighted average time to receive the bond's cash flows.
        """
        
        self.add_to_history('bot', response)
        
        # Generate a duration vs yield curve
        self.generate_duration_plot(coupon_rate, maturity, ytm, par)
    
    def show_yield_curve(self, _):
        """Show real yield curve data"""
        date, maturities, yields = self.calculator.get_real_yield_curve_data(date_index=5)
        
        if date is None:
            self.add_to_history('bot', "Could not load yield curve data. Make sure the necessary data files are available.")
            return
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, yields, 'b-o', label=f'Yield Curve on {date}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield')
        plt.title('Real Yield Curve Data')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        filename = 'yield_curve.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        response = f"""
Real Yield Curve Data for {date}:
-------------------------------
The yield curve shows interest rates for different maturities.
The plot has been saved as '{filename}'.

First few points:
- {maturities[0]:.2f} years: {yields[0]*100:.4f}%
- {maturities[1]:.2f} years: {yields[1]*100:.4f}%
- {maturities[2]:.2f} years: {yields[2]*100:.4f}%
...
- {maturities[-1]:.2f} years: {yields[-1]*100:.4f}%

Data source: mk.zero2.csv and mk.maturity.csv
        """
        
        self.add_to_history('bot', response)
    
    def show_zero_coupon(self, _):
        """Show zero coupon bond prices"""
        maturities, prices = self.calculator.get_zero_coupon_bond_prices()
        
        if maturities is None:
            self.add_to_history('bot', "Could not load zero-coupon bond data. Make sure the necessary data files are available.")
            return
        
        # Calculate yields
        maturities_yield, yields = self.calculator.calculate_yields_from_zero_prices()
        
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
        
        # Save the plot
        filename = 'zero_coupon.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        response = f"""
Zero-Coupon Bond Data:
--------------------
The plots show zero-coupon bond prices and their implied yields.
The plot has been saved as '{filename}'.

First few points:
- {maturities[0]:.2f} years: ${prices[0]:.2f}, {yields[0]*100:.4f}%
- {maturities[1]:.2f} years: ${prices[1]:.2f}, {yields[1]*100:.4f}%
- {maturities[2]:.2f} years: ${prices[2]:.2f}, {yields[2]*100:.4f}%
...
- {maturities[-1]:.2f} years: ${prices[-1]:.2f}, {yields[-1]*100:.4f}%

Data source: ZeroPrices.txt
        """
        
        self.add_to_history('bot', response)
    
    def show_treasury(self, _):
        """Show treasury yield curve data"""
        date, maturities, yields = self.calculator.get_treasury_yield_curve(date_index=0)
        
        if date is None:
            self.add_to_history('bot', "Could not load Treasury yield data. Make sure the necessary data files are available.")
            return
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, yields*100, 'r-o', label=f'Treasury Yield Curve on {date}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield (%)')
        plt.title('Treasury Yield Curve')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        filename = 'treasury_curve.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        response = f"""
Treasury Yield Curve Data for {date}:
----------------------------------
The Treasury yield curve shows interest rates on U.S. Treasury securities across different maturities.
The plot has been saved as '{filename}'.

Selected points:
"""
        
        # Add a few points to the response
        for i, (mat, yld) in enumerate(zip(maturities, yields)):
            if i < 3 or i > len(maturities) - 3:  # First 3 and last 3 points
                response += f"- {mat:.2f} years: {yld*100:.4f}%\n"
            elif i == 3:
                response += "...\n"
        
        response += "\nData source: treasury_yields.txt"
        
        self.add_to_history('bot', response)
    
    def plot_price_vs_yield(self, query):
        """Plot price vs yield curve"""
        coupon_rate = self.extract_param_value(query, "coupon", is_percentage=True, default=0.05)
        maturity = self.extract_param_value(query, "maturity", default=10)
        par = self.extract_param_value(query, "par", default=1000)
        
        self.generate_price_vs_yield_plot(coupon_rate, maturity, 0.05, par)
        
        response = f"""
Price vs. Yield Curve:
-------------------
Generated price vs. yield curve for a bond with:
- Par Value: ${par:.2f}
- Annual Coupon Rate: {coupon_rate*100:.2f}%
- Time to Maturity: {maturity:.2f} years

The plot shows how bond price changes with different yields.
The plot has been saved as 'price_vs_yield.png'.

This curve illustrates the inverse relationship between bond prices and yields.
        """
        
        self.add_to_history('bot', response)
    
    def generate_price_vs_yield_plot(self, coupon_rate, maturity, current_ytm=None, par=1000, current_price=None):
        """Generate a price vs yield curve plot"""
        # Create array of yields
        yields = np.linspace(0.01, 0.10, 100)  # 1% to 10%
        
        # Convert to semi-annual
        semiannual_coupon = coupon_rate * par / 2
        semiannual_yields = yields / 2
        
        # Calculate bond prices for each yield
        prices = [self.calculator.bondvalue(semiannual_coupon, maturity, y, par) for y in semiannual_yields]
        
        # Plot the price vs. yield curve
        plt.figure(figsize=(10, 6))
        plt.plot(yields * 100, prices, 'b-')
        
        # Add markers for current YTM and price if provided
        if current_ytm is not None:
            current_price = self.calculator.bondvalue(semiannual_coupon, maturity, current_ytm/2, par)
            plt.plot(current_ytm * 100, current_price, 'ro', markersize=8)
            plt.axvline(x=current_ytm*100, color='r', linestyle='--', 
                       label=f'Current YTM: {current_ytm*100:.2f}%')
            plt.axhline(y=current_price, color='g', linestyle='--', 
                       label=f'Current Price: ${current_price:.2f}')
        
        elif current_price is not None:
            # Need to calculate the YTM for this price
            semiannual_ytm = self.calculator.find_ytm_using_root(current_price, semiannual_coupon, maturity, par)
            annual_ytm = semiannual_ytm * 2
            plt.plot(annual_ytm * 100, current_price, 'ro', markersize=8)
            plt.axvline(x=annual_ytm*100, color='r', linestyle='--', 
                       label=f'Current YTM: {annual_ytm*100:.2f}%')
            plt.axhline(y=current_price, color='g', linestyle='--', 
                       label=f'Current Price: ${current_price:.2f}')
        
        plt.title(f'Bond Price vs. Yield Curve\n(Coupon: {coupon_rate*100}%, Maturity: {maturity} years)')
        plt.xlabel('Yield to Maturity (%)')
        plt.ylabel('Bond Price ($)')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        filename = 'price_vs_yield.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
    
    def generate_duration_plot(self, coupon_rate, maturity, current_ytm, par=1000):
        """Generate a duration vs yield or duration vs coupon plot"""
        # For duration vs yield
        yields = np.linspace(0.01, 0.10, 50)  # 1% to 10%
        
        # Convert to semi-annual
        semiannual_coupon = coupon_rate * par / 2
        semiannual_yields = yields / 2
        
        # Calculate durations for each yield
        durations = []
        for y in semiannual_yields:
            duration = self.calculator.calculate_bond_duration(semiannual_coupon, maturity, y, par)
            durations.append(duration)
        
        # Plot duration vs yield
        plt.figure(figsize=(10, 6))
        plt.plot(yields * 100, durations, 'b-')
        
        # Mark the current YTM
        current_duration = self.calculator.calculate_bond_duration(
            semiannual_coupon, maturity, current_ytm/2, par)
        plt.plot(current_ytm * 100, current_duration, 'ro', markersize=8)
        plt.axvline(x=current_ytm*100, color='r', linestyle='--', 
                   label=f'Current YTM: {current_ytm*100:.2f}%')
        
        plt.title(f'Bond Duration vs. Yield\n(Coupon: {coupon_rate*100}%, Maturity: {maturity} years)')
        plt.xlabel('Yield to Maturity (%)')
        plt.ylabel('Duration (years)')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        filename = 'duration_vs_yield.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
    
    def calculate_forward_rates(self, _):
        """Calculate and display forward rates"""
        date, maturities, spot_rates = self.calculator.get_real_yield_curve_data(date_index=5)
        
        if date is None:
            self.add_to_history('bot', "Could not load yield curve data. Make sure the necessary data files are available.")
            return
        
        # Calculate forward rates
        forward_rates = self.calculator.calculate_forward_rates(spot_rates, maturities)
        forward_maturities = [(maturities[i] + maturities[i-1])/2 for i in range(1, len(maturities))]
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, spot_rates, 'b-o', label='Spot Rates (Yield Curve)')
        plt.plot(forward_maturities, forward_rates, 'r-o', label='Forward Rates')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title(f'Yield Curve and Forward Rates on {date}')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        filename = 'forward_rates.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        response = f"""
Forward Rates Calculation:
-----------------------
Forward rates represent the interest rate for a future period implied by the current term structure.
The plot comparing spot rates and forward rates has been saved as '{filename}'.

Forward Rate Formula: Forward Rate(t₁,t₂) = (Spot Rate(t₂) × t₂ - Spot Rate(t₁) × t₁) / (t₂ - t₁)

Selected forward rates:
"""
        
        # Add a few points to the response
        for i, (mat, fwd) in enumerate(zip(forward_maturities, forward_rates)):
            if i < 3 or i > len(forward_rates) - 3:  # First 3 and last 3 points
                response += f"- Period around {mat:.2f} years: {fwd*100:.4f}%\n"
            elif i == 3:
                response += "...\n"
        
        response += "\nData source: Real yield curve data"
        
        self.add_to_history('bot', response)
    
    def calculate_correlation(self, query):
        """Calculate correlation between two stocks"""
        # Extract stock names from query
        stock1 = 'GM'  # Default
        stock2 = 'Ford'  # Default
        
        # Look for stock names in the query
        stock_match = re.search(r'correlation\s+(\w+)\s+(\w+)', query, re.IGNORECASE)
        if stock_match:
            stock1 = stock_match.group(1)
            stock2 = stock_match.group(2)
        
        # Get correlation data
        stock1_returns, stock2_returns, correlation = self.calculator.get_stock_bond_correlation(stock1, stock2)
        
        if stock1_returns is None:
            self.add_to_history('bot', f"Could not calculate correlation between {stock1} and {stock2}. Make sure the stock data is available.")
            return
        
        # Create plots
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
        
        # Save the plot
        filename = 'correlation.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        # Determine correlation strength
        strength = "strong positive" if correlation > 0.7 else \
                  "moderate positive" if correlation > 0.3 else \
                  "weak positive" if correlation > 0 else \
                  "weak negative" if correlation > -0.3 else \
                  "moderate negative" if correlation > -0.7 else \
                  "strong negative"
        
        response = f"""
Stock Correlation Analysis: {stock1} vs {stock2}
-------------------------------------------
Correlation Coefficient: {correlation:.4f}
Interpretation: {strength} correlation

The correlation analysis plots have been saved as '{filename}'.

This correlation represents the degree to which the returns of {stock1} and {stock2} move together.
A correlation of 1 means perfect positive correlation, 0 means no correlation, and -1 means perfect negative correlation.

Data source: Stock_Bond.csv
        """
        
        self.add_to_history('bot', response)
    
    def simulate_random_walk(self, query):
        """Simulate random walks for stock prices"""
        # Extract parameters from query
        stock = 'GM'  # Default
        stock_match = re.search(r'random walk\s+(\w+)', query, re.IGNORECASE)
        if stock_match:
            stock = stock_match.group(1)
        
        S0 = self.extract_param_value(query, "initial price", default=100)
        T = self.extract_param_value(query, "time horizon", default=5)
        num_paths = min(10, int(self.extract_param_value(query, "num paths", default=5)))
        
        # Get stock data for parameters
        stock_returns, _, _ = self.calculator.get_stock_bond_correlation(stock, 'Ford')
        
        if stock_returns is None:
            self.add_to_history('bot', f"Could not get stock data for {stock}. Using default parameters.")
            mu = 0.05  # Default drift
            sigma = 0.2  # Default volatility
        else:
            # Calculate parameters from real data
            mu = np.mean(stock_returns)
            sigma = np.std(stock_returns)
        
        # Simulate random walks
        times, paths = self.calculator.simulate_random_walk(S0=S0, mu=mu, sigma=sigma, T=T, num_paths=num_paths)
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        
        # Plot each path
        for i in range(paths.shape[0]):
            plt.plot(times, paths[i], label=f'Path {i+1}')
        
        # Plot mean and bounds
        mean_path = S0 * np.exp(mu * times)
        upper_bound = S0 * np.exp(mu * times + sigma * np.sqrt(times))
        lower_bound = S0 * np.exp(mu * times - sigma * np.sqrt(times))
        
        plt.plot(times, mean_path, 'k--', linewidth=2, label='Mean Path')
        plt.plot(times, upper_bound, 'r:', linewidth=2, label='Mean + 1 Std Dev')
        plt.plot(times, lower_bound, 'r:', linewidth=2, label='Mean - 1 Std Dev')
        
        plt.xlabel('Time (years)')
        plt.ylabel('Stock Price')
        plt.title(f'Random Walk Simulation for {stock}\n({num_paths} Paths, μ={mu:.6f}, σ={sigma:.6f})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = 'random_walk.png'
        plt.savefig(filename)
        plt.close()
        self.last_figure = filename
        
        response = f"""
Random Walk Simulation:
--------------------
Simulated {num_paths} random walk paths for {stock} with:
- Initial Price: ${S0:.2f}
- Time Horizon: {T} years
- Drift (μ): {mu:.6f}
- Volatility (σ): {sigma:.6f}

The simulation plot has been saved as '{filename}'.

This geometric Brownian motion model uses the formula:
dS = μS dt + σS dW

where dW is a random increment from a normal distribution.
The model is commonly used in finance to simulate stock prices and is the foundation for the Black-Scholes option pricing model.
        """
        
        self.add_to_history('bot', response)
    
    def list_data(self, _):
        """List available datasets"""
        datasets = self.calculator.list_available_data()
        
        if isinstance(datasets, str):
            self.add_to_history('bot', datasets)
            return
        
        response = "Available datasets:\n"
        for dataset in datasets:
            response += f"- {dataset}\n"
        
        response += "\nTo upload new data, use the 'upload' command."
        
        self.add_to_history('bot', response)
    
    def upload_data(self, query):
        """Upload a new dataset"""
        # Extract dataset name and file path
        upload_match = re.search(r'upload\s+(\w+)\s+(.+)', query, re.IGNORECASE)
        if not upload_match:
            self.add_to_history('bot', "Invalid upload command. Format: upload [dataset name] [file path]")
            return
        
        dataset_name = upload_match.group(1)
        file_path = upload_match.group(2).strip()
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.add_to_history('bot', f"File not found: {file_path}")
            return
        
        # Copy file to data directory
        os.makedirs(self.calculator.data_dir, exist_ok=True)
        dest_path = os.path.join(self.calculator.data_dir, os.path.basename(file_path))
        try:
            shutil.copy2(file_path, dest_path)
            success = self.calculator.add_dataset(dataset_name, dest_path)
            
            if success:
                self.add_to_history('bot', f"Successfully uploaded and added dataset '{dataset_name}' from {file_path}")
            else:
                self.add_to_history('bot', f"Error adding dataset. Make sure the file is in a supported format (CSV or TXT).")
        except Exception as e:
            self.add_to_history('bot', f"Error uploading dataset: {e}")
    
    def exit_chat(self, _):
        """Exit the chat interface"""
        self.add_to_history('bot', "Thank you for using the Bond Calculator Chat. Goodbye!")
        self.display_chat()
        sys.exit(0)

def main():
    """Main function to run the chat interface"""
    chat = ChatInterface()
    chat.start()

if __name__ == "__main__":
    main()