import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import newton, brentq
import os
import argparse
import pandas as pd

class BondCalculator:
    """
    A comprehensive bond calculator that implements solutions for various bond-related problems.
    """
    
    def __init__(self):
        """Initialize the BondCalculator class"""
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    # Questions 1-5: Bond Yield and Pricing Calculations
    
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
    
    def solve_question1(self):
        """
        Solve Question 1 about forward rate and bond pricing
        """
        # Given forward rate r(t) = 0.028 + 0.00042t
        
        def forward_rate(t):
            return 0.028 + 0.00042 * t
        
        # (a) Calculate the yield to maturity of a bond maturing in 20 years
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        ytm_20yr = calculate_spot_rate(20)
        
        # (b) Calculate the price of a par $1,000 zero-coupon bond maturing in 15 years
        spot_rate_15yr = calculate_spot_rate(15)
        price_15yr = 1000 * np.exp(-spot_rate_15yr * 15)
        
        # Create a plot to visualize the forward rate and spot rates
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 25, 100)
        forward_rates = [forward_rate(t) for t in t_values]
        spot_rates = [calculate_spot_rate(t) if t > 0 else forward_rate(0) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-', label='Forward Rate: r(t) = 0.028 + 0.00042t')
        plt.plot(t_values, spot_rates, 'r--', label='Spot Rate (YTM)')
        plt.axvline(x=20, color='g', linestyle=':', label='20-year maturity')
        plt.axhline(y=ytm_20yr, color='g', linestyle=':', label=f'20-year spot rate: {ytm_20yr:.4f}')
        plt.axvline(x=15, color='m', linestyle=':', label='15-year maturity')
        plt.axhline(y=spot_rate_15yr, color='m', linestyle=':', label=f'15-year spot rate: {spot_rate_15yr:.4f}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 1: Forward Rate and Spot Rates')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question1_forward_spot_rates.png'))
        plt.close()
        
        result = {
            'a_ytm_20yr': ytm_20yr,
            'b_price_15yr': price_15yr,
            'plot_path': os.path.join(self.output_dir, 'question1_forward_spot_rates.png')
        }
        
        return result
    
    def solve_question2(self):
        """
        Solve Question 2 about forward rate, yield curve, and bond returns
        """
        # Given forward rate r(t) = 0.04 + 0.0002t - 0.00003t^2
        
        def forward_rate(t):
            return 0.04 + 0.0002 * t - 0.00003 * (t**2)
        
        # (a) Calculate the yield to maturity of a bond maturing in 8 years
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        ytm_8yr = calculate_spot_rate(8)
        
        # (b) Calculate the price of a par $1,000 zero-coupon bond maturing in 5 years
        spot_rate_5yr = calculate_spot_rate(5)
        price_5yr = 1000 * np.exp(-spot_rate_5yr * 5)
        
        # (c) Plot the forward rate and yield curve
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 30, 300)
        forward_rates = [forward_rate(t) for t in t_values]
        spot_rates = [calculate_spot_rate(t) if t > 0 else forward_rate(0) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-', label='Forward Rate')
        plt.plot(t_values, spot_rates, 'r--', label='Yield Curve (Spot Rates)')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 2(c): Forward Rate and Yield Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'question2c_forward_yield_curves.png'))
        plt.close()
        
        # (d) Calculate the return from buying a 10-year zero-coupon bond and selling after 1 year
        # Calculate the initial price of the 10-year bond
        spot_rate_10yr = calculate_spot_rate(10)
        initial_price = 1000 * np.exp(-spot_rate_10yr * 10)
        
        # Calculate the price after 1 year (now a 9-year bond) assuming forward rate doesn't change
        spot_rate_9yr = calculate_spot_rate(9)
        price_after_1year = 1000 * np.exp(-spot_rate_9yr * 9)
        
        # Calculate the return
        return_1yr = (price_after_1year / initial_price) - 1
        
        result = {
            'a_ytm_8yr': ytm_8yr,
            'b_price_5yr': price_5yr,
            'c_plot_path': os.path.join(self.output_dir, 'question2c_forward_yield_curves.png'),
            'd_return_1yr': return_1yr
        }
        
        return result
    
    def solve_question3(self):
        """
        Solve Question 3 about coupon bond analysis
        """
        # A coupon bond has a coupon rate of 3% and a current yield of 2.8%
        
        coupon_rate = 0.03
        current_yield = 0.028
        
        # (a) Is the bond selling above or below par?
        # Current yield = Annual coupon payment / Bond price
        # If coupon rate > current yield, then price > par
        # If coupon rate < current yield, then price < par
        
        if coupon_rate > current_yield:
            above_par = True
            explanation_a = "The bond is selling above par because the coupon rate (3%) is greater than the current yield (2.8%). This means investors are willing to pay a premium for the bond's higher coupon payments."
        else:
            above_par = False
            explanation_a = "The bond is selling below par because the coupon rate (3%) is less than the current yield (2.8%). This means investors require a discount to achieve their desired yield."
        
        # (b) Is the yield to maturity above or below 2.8%?
        # For bonds selling above par, YTM < current yield
        # For bonds selling below par, YTM > current yield
        
        if above_par:
            ytm_comparison = "below"
            explanation_b = "The yield to maturity is below 2.8% (the current yield). For bonds selling above par, the yield to maturity must be less than the current yield because the bond will converge to par value at maturity, resulting in a capital loss that reduces the overall return."
        else:
            ytm_comparison = "above"
            explanation_b = "The yield to maturity is above 2.8% (the current yield). For bonds selling below par, the yield to maturity must be greater than the current yield because the bond will converge to par value at maturity, resulting in a capital gain that increases the overall return."
        
        result = {
            'a_above_par': above_par,
            'a_explanation': explanation_a,
            'b_ytm_comparison': ytm_comparison,
            'b_explanation': explanation_b
        }
        
        return result
    
    def solve_question4(self):
        """
        Solve Question 4 about forward rate and spot rate
        """
        # Given forward rate r(t) = 0.032 + 0.001t + 0.0002t^2
        
        def forward_rate(t):
            return 0.032 + 0.001 * t + 0.0002 * (t**2)
        
        # (a) Calculate the 5-year continuously compounded spot rate
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        spot_rate_5yr = calculate_spot_rate(5)
        
        # (b) Calculate the price of a zero-coupon bond that matures in 5 years
        price_5yr = 1000 * np.exp(-spot_rate_5yr * 5)
        
        # Create a plot to visualize the forward rate and spot rates
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 10, 100)
        forward_rates = [forward_rate(t) for t in t_values]
        spot_rates = [calculate_spot_rate(t) if t > 0 else forward_rate(0) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-', label='Forward Rate: r(t) = 0.032 + 0.001t + 0.0002t²')
        plt.plot(t_values, spot_rates, 'r--', label='Spot Rate (YTM)')
        plt.axvline(x=5, color='g', linestyle=':', label='5-year maturity')
        plt.axhline(y=spot_rate_5yr, color='g', linestyle=':', label=f'5-year spot rate: {spot_rate_5yr:.4f}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 4: Forward Rate and Spot Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question4_forward_spot_rates.png'))
        plt.close()
        
        result = {
            'a_spot_rate_5yr': spot_rate_5yr,
            'b_price_5yr': price_5yr,
            'plot_path': os.path.join(self.output_dir, 'question4_forward_spot_rates.png')
        }
        
        return result
    
    def solve_question5(self):
        """
        Solve Question 5 about bond pricing with spot rates
        """
        # Given spot rates (semiannually compounded)
        spot_rates = {
            0.5: 0.025,  # 1/2-year spot rate
            1.0: 0.028,  # 1-year spot rate
            1.5: 0.032,  # 1.5-year spot rate
            2.0: 0.033   # 2-year spot rate
        }
        
        # Convert semiannually compounded rates to continuously compounded rates
        spot_rates_continuous = {t: 2 * np.log(1 + rate) for t, rate in spot_rates.items()}
        
        # Bond details
        par = 1000
        coupon = 35  # Semiannual coupon payment
        
        # Calculate the present value of each cash flow
        pv_6months = coupon * np.exp(-spot_rates_continuous[0.5] * 0.5)
        pv_12months = coupon * np.exp(-spot_rates_continuous[1.0] * 1.0)
        pv_18months = coupon * np.exp(-spot_rates_continuous[1.5] * 1.5)
        pv_24months = (coupon + par) * np.exp(-spot_rates_continuous[2.0] * 2.0)
        
        # Total bond price
        bond_price = pv_6months + pv_12months + pv_18months + pv_24months
        
        # Create a visualization of the cash flows and their present values
        plt.figure(figsize=(10, 6))
        
        # Cash flows
        cash_flows = [coupon, coupon, coupon, coupon + par]
        times = [0.5, 1.0, 1.5, 2.0]
        present_values = [pv_6months, pv_12months, pv_18months, pv_24months]
        
        plt.bar(times, cash_flows, width=0.2, alpha=0.7, label='Cash Flows')
        plt.bar([t + 0.2 for t in times], present_values, width=0.2, alpha=0.7, label='Present Values')
        
        for i, (t, cf, pv) in enumerate(zip(times, cash_flows, present_values)):
            plt.text(t, cf + 10, f'${cf:.2f}', ha='center')
            plt.text(t + 0.2, pv + 10, f'${pv:.2f}', ha='center')
        
        plt.axhline(y=bond_price, color='r', linestyle='-', label=f'Bond Price: ${bond_price:.2f}')
        plt.xlabel('Time (years)')
        plt.ylabel('Amount ($)')
        plt.title('Question 5: Bond Cash Flows and Present Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'question5_bond_cash_flows.png'))
        plt.close()
        
        result = {
            'bond_price': bond_price,
            'cash_flows': {
                '6_months': {'cash_flow': coupon, 'present_value': pv_6months},
                '12_months': {'cash_flow': coupon, 'present_value': pv_12months},
                '18_months': {'cash_flow': coupon, 'present_value': pv_18months},
                '24_months': {'cash_flow': coupon + par, 'present_value': pv_24months}
            },
            'plot_path': os.path.join(self.output_dir, 'question5_bond_cash_flows.png')
        }
        
        return result
    
    # Questions 6-8: Yield to Maturity and Bond Pricing
    
    def solve_question6(self):
        """
        Solve Question 6 about finding coupon payment given yield to maturity
        """
        # The yield to maturity is 0.035 on a par $1,000 bond selling at $950.10 and maturing in 5 years
        
        price = 950.10
        ytm = 0.035
        T = 5
        par = 1000
        
        # Find the coupon payment
        # For a coupon bond: Price = c/r + (par - c/r) * (1 + r)^(-2T)
        # Solving for c: c = r * (price - par * (1 + r)^(-2T)) / (1 - (1 + r)^(-2T))
        
        discount_factor = (1 + ytm) ** (-2 * T)
        coupon = ytm * (price - par * discount_factor) / (1 - discount_factor)
        
        # Annual coupon payment
        annual_coupon = coupon * 2
        
        # Coupon rate
        coupon_rate = annual_coupon / par
        
        # Verify the calculation
        calculated_price = self.bondvalue(coupon, T, ytm, par)
        
        # Create a plot to visualize the relationship between coupon and price
        plt.figure(figsize=(10, 6))
        coupon_values = np.linspace(15, 40, 100)
        prices = [self.bondvalue(c, T, ytm, par) for c in coupon_values]
        
        plt.plot(coupon_values, prices, 'b-')
        plt.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
        plt.axvline(x=coupon, color='g', linestyle='--', label=f'Coupon = ${coupon:.2f}')
        plt.xlabel('Semiannual Coupon Payment ($)')
        plt.ylabel('Bond Price ($)')
        plt.title('Question 6: Bond Price vs. Coupon Payment')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question6_bond_coupon_plot.png'))
        plt.close()
        
        result = {
            'coupon': coupon,
            'annual_coupon': annual_coupon,
            'coupon_rate': coupon_rate,
            'verification_price': calculated_price,
            'plot_path': os.path.join(self.output_dir, 'question6_bond_coupon_plot.png')
        }
        
        return result
    
    def solve_question7(self):
        """
        Solve Question 7 about yield to maturity of a 30-year par $1,000 bond
        """
        # A 30-year par $1,000 bond with coupon payments of $40 that is selling at $1,200
        
        price = 1200
        coupon = 40
        T = 30
        par = 1000
        
        # Find the yield to maturity using root-finding method
        def bond_price_diff(ytm):
            calculated_price = self.bondvalue(coupon, T, ytm, par)
            return calculated_price - price
        
        # Use a numerical method to find the yield to maturity
        ytm = brentq(bond_price_diff, 0.001, 0.1)
        
        # Annual yield to maturity
        annual_ytm = ytm * 2
        
        # Verify the calculation
        calculated_price = self.bondvalue(coupon, T, ytm, par)
        
        # Create a plot to visualize the relationship between yield and price
        plt.figure(figsize=(10, 6))
        ytm_values = np.linspace(0.01, 0.05, 100)
        prices = [self.bondvalue(coupon, T, r, par) for r in ytm_values]
        
        plt.plot(ytm_values, prices, 'b-')
        plt.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
        plt.axvline(x=ytm, color='g', linestyle='--', label=f'YTM = {ytm:.4f}')
        plt.xlabel('Yield to Maturity (semiannual rate)')
        plt.ylabel('Bond Price ($)')
        plt.title('Question 7: Bond Price vs. Yield to Maturity')
        plt.grid(True)
(Content truncated due to size limit. Use line ranges to read in chunks)