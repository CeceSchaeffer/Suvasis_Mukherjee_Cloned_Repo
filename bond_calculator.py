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
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question7_bond_yield_plot.png'))
        plt.close()
        
        result = {
            'ytm': ytm,
            'annual_ytm': annual_ytm,
            'verification_price': calculated_price,
            'plot_path': os.path.join(self.output_dir, 'question7_bond_yield_plot.png')
        }
        
        return result
    
    def solve_question8(self):
        """
        Solve Question 8 about yield to maturity of a 20-year par $1,000 bond
        """
        # A 20-year par $1,000 bond with semiannual coupon payments of $35 that is selling at $1,050
        
        price = 1050
        coupon = 35
        T = 20
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
        plt.title('Question 8: Bond Price vs. Yield to Maturity')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question8_bond_yield_plot.png'))
        plt.close()
        
        result = {
            'ytm': ytm,
            'annual_ytm': annual_ytm,
            'verification_price': calculated_price,
            'plot_path': os.path.join(self.output_dir, 'question8_bond_yield_plot.png')
        }
        
        return result
    
    # Questions 9-13: More Bond Calculations
    
    def solve_question9(self):
        """
        Solve Question 9 about bond pricing with a given yield
        """
        # A coupon bond with a par value of $1,000 and a 10-year maturity pays semiannual coupons of $21
        
        coupon = 21
        T = 10
        par = 1000
        ytm = 0.04 / 2  # 4% annual yield, semiannual compounding
        
        # (a) Calculate the price of the bond
        price = self.bondvalue(coupon, T, ytm, par)
        
        # (b) Determine if the bond is selling above or below par
        if price > par:
            above_par = True
            explanation = "The bond is selling above par because the coupon rate (4.2% annually) is higher than the yield to maturity (4%)."
        elif price < par:
            above_par = False
            explanation = "The bond is selling below par because the coupon rate (4.2% annually) is lower than the yield to maturity (4%)."
        else:
            above_par = None
            explanation = "The bond is selling at par because the coupon rate equals the yield to maturity."
        
        # Create a plot to visualize the relationship between yield and price
        plt.figure(figsize=(10, 6))
        ytm_values = np.linspace(0.01, 0.05, 100)
        prices = [self.bondvalue(coupon, T, r, par) for r in ytm_values]
        
        plt.plot(ytm_values, prices, 'b-')
        plt.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
        plt.axhline(y=par, color='g', linestyle='--', label=f'Par = ${par}')
        plt.axvline(x=ytm, color='m', linestyle='--', label=f'YTM = {ytm*2:.4f}')
        plt.xlabel('Yield to Maturity (semiannual rate)')
        plt.ylabel('Bond Price ($)')
        plt.title('Question 9: Bond Price vs. Yield to Maturity')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question9_bond_price_plot.png'))
        plt.close()
        
        result = {
            'a_price': price,
            'b_above_par': above_par,
            'b_explanation': explanation,
            'plot_path': os.path.join(self.output_dir, 'question9_bond_price_plot.png')
        }
        
        return result
    
    def solve_question10(self):
        """
        Solve Question 10 about yield to maturity and current yield
        """
        # A coupon bond with a par value of $1,000 and a maturity of 7 years is selling for $1,040
        # The semiannual coupon payments are $23
        
        price = 1040
        coupon = 23
        T = 7
        par = 1000
        
        # (a) Find the yield to maturity
        def bond_price_diff(ytm):
            calculated_price = self.bondvalue(coupon, T, ytm, par)
            return calculated_price - price
        
        # Use a numerical method to find the yield to maturity
        ytm = brentq(bond_price_diff, 0.001, 0.1)
        
        # Annual yield to maturity
        annual_ytm = ytm * 2
        
        # (b) Calculate the current yield
        annual_coupon = coupon * 2
        current_yield = annual_coupon / price
        
        # (c) Compare yield to maturity and current yield
        if ytm * 2 < current_yield:
            comparison = "less than"
            explanation = "The yield to maturity is less than the current yield because the bond is selling above par and will converge to par at maturity, creating a capital loss."
        elif ytm * 2 > current_yield:
            comparison = "greater than"
            explanation = "The yield to maturity is greater than the current yield because the bond is selling below par and will converge to par at maturity, creating a capital gain."
        else:
            comparison = "equal to"
            explanation = "The yield to maturity equals the current yield, which would occur if the bond is selling at par."
        
        # Create a plot to visualize the relationship between yield and price
        plt.figure(figsize=(10, 6))
        ytm_values = np.linspace(0.01, 0.05, 100)
        prices = [self.bondvalue(coupon, T, r, par) for r in ytm_values]
        
        plt.plot(ytm_values, prices, 'b-')
        plt.axhline(y=price, color='r', linestyle='--', label=f'Price = ${price:.2f}')
        plt.axvline(x=ytm, color='g', linestyle='--', label=f'YTM = {ytm*2:.4f}')
        plt.axhline(y=par, color='m', linestyle=':', label=f'Par = ${par}')
        plt.xlabel('Yield to Maturity (semiannual rate)')
        plt.ylabel('Bond Price ($)')
        plt.title('Question 10: Bond Price vs. Yield to Maturity')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question10_bond_yield_plot.png'))
        plt.close()
        
        result = {
            'a_ytm': ytm,
            'a_annual_ytm': annual_ytm,
            'b_current_yield': current_yield,
            'c_comparison': comparison,
            'c_explanation': explanation,
            'plot_path': os.path.join(self.output_dir, 'question10_bond_yield_plot.png')
        }
        
        return result
    
    def solve_question11(self):
        """
        Solve Question 11 about zero-coupon bond pricing with forward rate
        """
        # Given forward rate r(t) = 0.033 + 0.0012t
        
        def forward_rate(t):
            return 0.033 + 0.0012 * t
        
        # Calculate the value of a par $100 zero-coupon bond with a maturity of 15 years
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        spot_rate_15yr = calculate_spot_rate(15)
        price = 100 * np.exp(-spot_rate_15yr * 15)
        
        # Create a plot to visualize the forward rate and spot rates
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 20, 100)
        forward_rates = [forward_rate(t) for t in t_values]
        spot_rates = [calculate_spot_rate(t) if t > 0 else forward_rate(0) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-', label='Forward Rate: r(t) = 0.033 + 0.0012t')
        plt.plot(t_values, spot_rates, 'r--', label='Spot Rate (YTM)')
        plt.axvline(x=15, color='g', linestyle=':', label='15-year maturity')
        plt.axhline(y=spot_rate_15yr, color='g', linestyle=':', label=f'15-year spot rate: {spot_rate_15yr:.4f}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 11: Forward Rate and Spot Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question11_forward_spot_rates.png'))
        plt.close()
        
        result = {
            'spot_rate_15yr': spot_rate_15yr,
            'price': price,
            'plot_path': os.path.join(self.output_dir, 'question11_forward_spot_rates.png')
        }
        
        return result
    
    def solve_question12(self):
        """
        Solve Question 12 about bond return with changing forward rate
        """
        # Initial forward rate r(t) = 0.04 + 0.001t
        # After 6 months, forward rate r(t) = 0.03 + 0.0013t
        
        def forward_rate_initial(t):
            return 0.04 + 0.001 * t
        
        def forward_rate_after(t):
            return 0.03 + 0.0013 * t
        
        # Calculate spot rates
        def calculate_spot_rate(forward_rate_func, T):
            integral, _ = integrate.quad(forward_rate_func, 0, T)
            return integral / T
        
        # Calculate the initial price of the 8-year zero-coupon bond
        spot_rate_initial = calculate_spot_rate(forward_rate_initial, 8)
        initial_price = 100 * np.exp(-spot_rate_initial * 8)
        
        # Calculate the price after 6 months (now a 7.5-year bond) with the new forward rate
        spot_rate_after = calculate_spot_rate(forward_rate_after, 7.5)
        price_after = 100 * np.exp(-spot_rate_after * 7.5)
        
        # Calculate the return
        return_value = (price_after / initial_price) - 1
        
        # Create a plot to visualize the forward rates before and after
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 10, 100)
        forward_rates_initial = [forward_rate_initial(t) for t in t_values]
        forward_rates_after = [forward_rate_after(t) for t in t_values]
        
        plt.plot(t_values, forward_rates_initial, 'b-', label='Initial Forward Rate: r(t) = 0.04 + 0.001t')
        plt.plot(t_values, forward_rates_after, 'r--', label='New Forward Rate: r(t) = 0.03 + 0.0013t')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 12: Forward Rates Before and After')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question12_forward_rates.png'))
        plt.close()
        
        result = {
            'initial_spot_rate': spot_rate_initial,
            'initial_price': initial_price,
            'new_spot_rate': spot_rate_after,
            'price_after': price_after,
            'return': return_value,
            'plot_path': os.path.join(self.output_dir, 'question12_forward_rates.png')
        }
        
        return result
    
    def solve_question13(self):
        """
        Solve Question 13 about yield to maturity with piecewise forward rate
        """
        # Given forward rate r(t) = 0.03 + 0.001t - 0.0002(t-10)+
        
        def forward_rate(t):
            positive_part = max(0, t - 10)
            return 0.03 + 0.001 * t - 0.0002 * positive_part
        
        # Calculate the yield to maturity on a 20-year zero-coupon bond
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        ytm_20yr = calculate_spot_rate(20)
        
        # Create a plot to visualize the forward rate
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 25, 250)
        forward_rates = [forward_rate(t) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-')
        plt.axvline(x=10, color='r', linestyle='--', label='t = 10 (breakpoint)')
        plt.axvline(x=20, color='g', linestyle=':', label='20-year maturity')
        plt.axhline(y=ytm_20yr, color='g', linestyle=':', label=f'20-year YTM: {ytm_20yr:.4f}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 13: Forward Rate with Positive Part Function')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question13_forward_rate.png'))
        plt.close()
        
        result = {
            'ytm_20yr': ytm_20yr,
            'plot_path': os.path.join(self.output_dir, 'question13_forward_rate.png')
        }
        
        return result
    
    # Questions 14-20: Already implemented in the original code
    
    def bond_duration_derivative(self):
        """
        Solve the bond duration derivative problem (Question 15)
        """
        proof_steps = """
        To prove the relationship between the derivative and duration, I'll work through the steps:

        1. Start with the left side of the equation:
           d/dδ [∑(i=1 to N) Ci exp(-Ti(yTi + δ))]|δ=0

        2. Apply the derivative with respect to δ:
           d/dδ [∑(i=1 to N) Ci exp(-Ti(yTi + δ))] = ∑(i=1 to N) Ci · d/dδ[exp(-Ti(yTi + δ))]

        3. Using the chain rule:
           d/dδ[exp(-Ti(yTi + δ))] = exp(-Ti(yTi + δ)) · d/dδ[-Ti(yTi + δ)] = exp(-Ti(yTi + δ)) · (-Ti)

        4. Substituting back:
           ∑(i=1 to N) Ci · exp(-Ti(yTi + δ)) · (-Ti)

        5. Evaluating at δ = 0:
           ∑(i=1 to N) Ci · exp(-TiyTi) · (-Ti) = -∑(i=1 to N) Ti · Ci · exp(-TiyTi)

        6. Recall that NPVi = Ci exp(-TiyTi) and ωi = NPVi / ∑NPVj

        7. Rewriting:
           -∑(i=1 to N) Ti · Ci · exp(-TiyTi) = -∑(i=1 to N) Ti · NPVi

        8. Factoring out the total NPV:
           = -[∑(i=1 to N) Ti · (NPVi / ∑NPVj) · ∑NPVj]
           = -[∑(i=1 to N) Ti · ωi] · ∑NPVj
           = -DUR · ∑(i=1 to N) Ci exp(-TiyTi)

        This proves the relationship:
        d/dδ [∑(i=1 to N) Ci exp(-Ti(yTi + δ))]|δ=0 = -DUR ∑(i=1 to N) Ci exp{-TiyTi}

        This result verifies Equation (3.31) in the textbook, showing that the duration is related to the sensitivity of the bond price to changes in yield.
        """
        
        return proof_steps
    
    def zero_coupon_bond_with_yield_curve(self):
        """
        Solve the zero-coupon bond with yield curve problem (Question 16)
        """
        # (a) Calculate the price of a par-$1,000 zero-coupon bond with a maturity of 10 years
        # Given yield curve: Y_T = 0.04 + 0.001 * T
        
        def yield_curve(T):
            return 0.04 + 0.001 * T
        
        # For a 10-year zero-coupon bond
        T = 10
        y_T = yield_curve(T)
        
        # Price = Par * exp(-y_T * T)
        price_a = 1000 * np.exp(-y_T * T)
        
        # (b) Calculate the return after 1 year if yield curve changes to Y_T = 0.042 + 0.001 * T
        
        def new_yield_curve(T):
            return 0.042 + 0.001 * T
        
        # After 1 year, the bond has 9 years to maturity
        T_new = 9
        y_T_new = new_yield_curve(T_new)
        
        # New price
        price_b = 1000 * np.exp(-y_T_new * T_new)
        
        # Calculate return
        return_b = (price_b / price_a) - 1
        
        # Create a plot to visualize the yield curves
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 15, 100)
        plt.plot(t_values, [yield_curve(t) for t in t_values], 'b-', label='Initial Yield Curve: Y_T = 0.04 + 0.001T')
        plt.plot(t_values, [new_yield_curve(t) for t in t_values], 'r--', label='New Yield Curve: Y_T = 0.042 + 0.001T')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield')
        plt.title('Question 16: Yield Curves')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question16_yield_curves.png'))
        plt.close()
        
        result = {
            'a_price': price_a,
            'b_price_after_1yr': price_b,
            'b_return': return_b,
            'plot_path': os.path.join(self.output_dir, 'question16_yield_curves.png')
        }
        
        return result
    
    def coupon_bond_analysis(self):
        """
        Solve the coupon bond analysis problem (Question 17)
        """
        # (a) Is the bond selling above or below par?
        # Current yield = Annual coupon payment / Bond price
        # If coupon rate > current yield, then price > par
        # If coupon rate < current yield, then price < par
        
        coupon_rate = 0.03
        current_yield = 0.028
        
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
    
    def forward_rate_and_spot_rate(self):
        """
        Solve the forward rate and spot rate problem (Question 18)
        """
        # (a) Calculate the 5-year spot rate
        
        def forward_rate(t):
            return 0.03 + 0.001 * t + 0.0002 * (t**2)
        
        # Spot rate is the average of forward rates from 0 to T
        def calculate_spot_rate(T):
            integral, _ = integrate.quad(forward_rate, 0, T)
            return integral / T
        
        # Calculate 5-year spot rate
        spot_rate_5yr = calculate_spot_rate(5)
        
        # (b) Calculate the price of a zero-coupon bond that matures in 5 years
        # Price = Par * exp(-spot_rate * T)
        price_5yr = 1000 * np.exp(-spot_rate_5yr * 5)
        
        # Create a plot to visualize the forward rate and spot rates
        plt.figure(figsize=(10, 6))
        t_values = np.linspace(0, 10, 100)
        forward_rates = [forward_rate(t) for t in t_values]
        spot_rates = [calculate_spot_rate(t) if t > 0 else forward_rate(0) for t in t_values]
        
        plt.plot(t_values, forward_rates, 'b-', label='Forward Rate: r(t) = 0.03 + 0.001t + 0.0002t²')
        plt.plot(t_values, spot_rates, 'r--', label='Spot Rate (YTM)')
        plt.axvline(x=5, color='g', linestyle=':', label='5-year maturity')
        plt.axhline(y=spot_rate_5yr, color='g', linestyle=':', label=f'5-year spot rate: {spot_rate_5yr:.4f}')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Rate')
        plt.title('Question 18: Forward Rate and Spot Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'question18_forward_spot_rates.png'))
        plt.close()
        
        result = {
            'a_spot_rate': spot_rate_5yr,
            'b_price': price_5yr,
            'plot_path': os.path.join(self.output_dir, 'question18_forward_spot_rates.png')
        }
        
        return result
    
    def bond_pricing_with_spot_rates(self):
        """
        Solve the bond pricing with spot rates problem (Question 19)
        """
        # Given spot rates (semiannually compounded)
        spot_rates = {
            0.5: 0.025,  # 1/2-year spot rate
            1.0: 0.029,  # 1-year spot rate
            1.5: 0.031,  # 1.5-year spot rate
            2.0: 0.035   # 2-year spot rate
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
        plt.title('Question 19: Bond Cash Flows and Present Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'question19_bond_cash_flows.png'))
        plt.close()
        
        result = {
            'bond_price': bond_price,
            'cash_flows': {
                '6_months': {'cash_flow': coupon, 'present_value': pv_6months},
                '12_months': {'cash_flow': coupon, 'present_value': pv_12months},
                '18_months': {'cash_flow': coupon, 'present_value': pv_18months},
                '24_months': {'cash_flow': coupon + par, 'present_value': pv_24months}
            },
            'plot_path': os.path.join(self.output_dir, 'question19_bond_cash_flows.png')
        }
        
        return result
    
    def calculate_spot_rates_from_prices(self):
        """
        Solve the calculate spot rates from bond prices problem (Question 20)
        """
        # Given zero-coupon bond prices
        bond_prices = {
            0.5: 980.39,  # 0.5-year bond price
            1.0: 957.41,  # 1-year bond price
            1.5: 923.18,  # 1.5-year bond price
            2.0: 888.49   # 2-year bond price
        }
        
        # Calculate semiannual spot rates
        # For zero-coupon bonds: Price = Par / (1 + r/2)^(2t)
        # Solving for r: r = 2 * [(Par/Price)^(1/(2t)) - 1]
        
        par = 1000
        spot_rates = {}
        
        for t, price in bond_prices.items():
            periods = 2 * t  # Number of semiannual periods
            spot_rates[t] = 2 * ((par / price) ** (1 / periods) - 1)
        
        # Create a visualization of the spot rate curve
        plt.figure(figsize=(10, 6))
        
        times = list(spot_rates.keys())
        rates = [spot_rates[t] * 100 for t in times]  # Convert to percentage
        
        plt.plot(times, rates, 'bo-', linewidth=2)
        
        for i, (t, r) in enumerate(zip(times, rates)):
            plt.text(t, r + 0.1, f'{r:.2f}%', ha='center')
        
        plt.xlabel('Maturity (years)')
        plt.ylabel('Spot Rate (%)')
        plt.title('Question 20: Spot Rate Curve Derived from Zero-Coupon Bond Prices')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'question20_spot_rate_curve.png'))
        plt.close()
        
        result = {
            'spot_rates': {t: rate * 100 for t, rate in spot_rates.items()},  # Convert to percentage
            'plot_path': os.path.join(self.output_dir, 'question20_spot_rate_curve.png')
        }
        
        return result
    
    def zero_coupon_bond_investment_analysis(self):
        """
        Solve the zero-coupon bond investment analysis problem (Question 14)
        """
        # Current spot rates (semiannual compounding)
        current_spot_rates = {
            1: 0.031,  # 1-year spot rate
            2: 0.035,  # 2-year spot rate
            3: 0.040,  # 3-year spot rate
            4: 0.042,  # 4-year spot rate
            5: 0.043   # 5-year spot rate
        }
        
        # Increased spot rates (each increased by 0.005)
        increased_spot_rates = {year: rate + 0.005 for year, rate in current_spot_rates.items()}
        
        # (a) Calculate current prices of 1-, 3-, and 5-year zero-coupon bonds
        def calculate_zero_coupon_price(spot_rate, years, par=1000):
            periods = years * 2  # Semiannual compounding
            rate_per_period = spot_rate / 2
            price = par / ((1 + rate_per_period) ** periods)
            return price
        
        price_1yr_current = calculate_zero_coupon_price(current_spot_rates[1], 1)
        price_3yr_current = calculate_zero_coupon_price(current_spot_rates[3], 3)
        price_5yr_current = calculate_zero_coupon_price(current_spot_rates[5], 5)
        
        # (b) Calculate prices after 1 year if spot rates remain unchanged
        price_1yr_after_unchanged = 1000  # Matured to par value
        price_3yr_after_unchanged = calculate_zero_coupon_price(current_spot_rates[2], 2)  # Now a 2-year bond
        price_5yr_after_unchanged = calculate_zero_coupon_price(current_spot_rates[4], 4)  # Now a 4-year bond
        
        # (c) Calculate prices after 1 year if spot rates increase by 0.005
        price_1yr_after_increased = 1000  # Matured to par value
        price_3yr_after_increased = calculate_zero_coupon_price(increased_spot_rates[2], 2)  # Now a 2-year bond
        price_5yr_after_increased = calculate_zero_coupon_price(increased_spot_rates[4], 4)  # Now a 4-year bond
        
        # (d) Calculate returns if spot rates increase by 0.005
        def calculate_one_year_return(initial_price, price_after_one_year):
            return (price_after_one_year / initial_price - 1) * 100
        
        return_1yr_increased = calculate_one_year_return(price_1yr_current, price_1yr_after_increased)
        return_3yr_increased = calculate_one_year_return(price_3yr_current, price_3yr_after_increased)
        return_5yr_increased = calculate_one_year_return(price_5yr_current, price_5yr_after_increased)
        
        # (e) Calculate returns if spot rates remain unchanged
        return_1yr_unchanged = calculate_one_year_return(price_1yr_current, price_1yr_after_unchanged)
        return_3yr_unchanged = calculate_one_year_return(price_3yr_current, price_3yr_after_unchanged)
        return_5yr_unchanged = calculate_one_year_return(price_5yr_current, price_5yr_after_unchanged)
        
        # Create a bar chart comparing returns under both scenarios
        plt.figure(figsize=(10, 6))
        maturities = ['1-year', '3-year', '5-year']
        returns_unchanged = [return_1yr_unchanged, return_3yr_unchanged, return_5yr_unchanged]
        returns_increased = [return_1yr_increased, return_3yr_increased, return_5yr_increased]
        
        x = np.arange(len(maturities))
        width = 0.35
        
        plt.bar(x - width/2, returns_unchanged, width, label='Unchanged Rates')
        plt.bar(x + width/2, returns_increased, width, label='Increased Rates')
        
        plt.xlabel('Bond Maturity')
        plt.ylabel('1-Year Return (%)')
        plt.title('Question 14: Comparison of 1-Year Returns by Maturity and Rate Scenario')
        plt.xticks(x, maturities)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(self.output_dir, 'question14_bond_returns_comparison.png'))
        plt.close()
        
        result = {
            'current_prices': {
                '1_year': price_1yr_current,
                '3_year': price_3yr_current,
                '5_year': price_5yr_current
            },
            'prices_after_1yr_unchanged': {
                '1_year': price_1yr_after_unchanged,
                '3_year': price_3yr_after_unchanged,
                '5_year': price_5yr_after_unchanged
            },
            'prices_after_1yr_increased': {
                '1_year': price_1yr_after_increased,
                '3_year': price_3yr_after_increased,
                '5_year': price_5yr_after_increased
            },
            'returns_unchanged': {
                '1_year': return_1yr_unchanged,
                '3_year': return_3yr_unchanged,
                '5_year': return_5yr_unchanged
            },
            'returns_increased': {
                '1_year': return_1yr_increased,
                '3_year': return_3yr_increased,
                '5_year': return_5yr_increased
            },
            'best_maturity_unchanged': '3-year' if return_3yr_unchanged > max(return_1yr_unchanged, return_5yr_unchanged) else ('1-year' if return_1yr_unchanged > return_5yr_unchanged else '5-year'),
            'best_maturity_increased': '3-year' if return_3yr_increased > max(return_1yr_increased, return_5yr_increased) else ('1-year' if return_1yr_increased > return_5yr_increased else '5-year'),
            'plot_path': os.path.join(self.output_dir, 'question14_bond_returns_comparison.png')
        }
        
        return result
    
    def run_all_calculations(self):
        """
        Run all bond calculations and return the results
        """
        results = {
            # Questions 1-5
            'question1': self.solve_question1(),
            'question2': self.solve_question2(),
            'question3': self.solve_question3(),
            'question4': self.solve_question4(),
            'question5': self.solve_question5(),
            
            # Questions 6-8
            'question6': self.solve_question6(),
            'question7': self.solve_question7(),
            'question8': self.solve_question8(),
            
            # Questions 9-13
            'question9': self.solve_question9(),
            'question10': self.solve_question10(),
            'question11': self.solve_question11(),
            'question12': self.solve_question12(),
            'question13': self.solve_question13(),
            
            # Questions 14-20
            'question14': self.zero_coupon_bond_investment_analysis(),
            'question15': self.bond_duration_derivative(),
            'question16': self.zero_coupon_bond_with_yield_curve(),
            'question17': self.coupon_bond_analysis(),
            'question18': self.forward_rate_and_spot_rate(),
            'question19': self.bond_pricing_with_spot_rates(),
            'question20': self.calculate_spot_rates_from_prices()
        }
        
        return results
    
    def generate_report(self, results):
        """
        Generate a comprehensive report of all bond calculations
        """
        report = """
        # Bond Calculator Results
        
        ## Questions 1-5: Bond Yield and Pricing Calculations
        
        ### Question 1: Forward Rate and Bond Pricing
        
        Given forward rate r(t) = 0.028 + 0.00042t
        
        #### (a) Yield to maturity of a bond maturing in 20 years:
        {:.6f} or {:.4f}%
        
        #### (b) Price of a par $1,000 zero-coupon bond maturing in 15 years:
        ${:.2f}
        
        ### Question 2: Forward Rate, Yield Curve, and Bond Returns
        
        Given forward rate r(t) = 0.04 + 0.0002t - 0.00003t²
        
        #### (a) Yield to maturity of a bond maturing in 8 years:
        {:.6f} or {:.4f}%
        
        #### (b) Price of a par $1,000 zero-coupon bond maturing in 5 years:
        ${:.2f}
        
        #### (d) Return from buying a 10-year zero-coupon bond and selling after 1 year:
        {:.6f} or {:.4f}%
        
        ### Question 3: Coupon Bond Analysis
        
        A coupon bond has a coupon rate of 3% and a current yield of 2.8%.
        
        #### (a) Is the bond selling above or below par?
        {}
        
        #### (b) Is the yield to maturity above or below 2.8%?
        {}
        
        ### Question 4: Forward Rate and Spot Rate
        
        Given forward rate r(t) = 0.032 + 0.001t + 0.0002t²
        
        #### (a) 5-year continuously compounded spot rate:
        {:.6f} or {:.4f}%
        
        #### (b) Price of a zero-coupon bond that matures in 5 years:
        ${:.2f}
        
        ### Question 5: Bond Pricing with Spot Rates
        
        The 1/2-, 1-, 1.5-, and 2-year semiannually compounded spot rates are 0.025, 0.028, 0.032, and 0.033, respectively.
        
        #### Price of 2-year coupon bond with semiannual payments of $35:
        ${:.2f}
        
        ## Questions 6-8: Yield to Maturity and Bond Pricing
        
        ### Question 6: Finding Coupon Payment Given Yield to Maturity
        
        The yield to maturity is 0.035 on a par $1,000 bond selling at $950.10 and maturing in 5 years.
        
        #### Coupon payment:
        ${:.2f} (semiannual)
        
        #### Annual coupon payment:
        ${:.2f}
        
        #### Coupon rate:
        {:.4f}%
        
        ### Question 7: Yield to Maturity of a 30-year Bond
        
        A 30-year par $1,000 bond with coupon payments of $40 that is selling at $1,200.
        
        #### Yield to maturity:
        {:.6f} (semiannual rate) or {:.4f}% (annual rate)
        
        ### Question 8: Yield to Maturity of a 20-year Bond
        
        A 20-year par $1,000 bond with semiannual coupon payments of $35 that is selling at $1,050.
        
        #### Yield to maturity:
        {:.6f} (semiannual rate) or {:.4f}% (annual rate)
        
        ## Questions 9-13: More Bond Calculations
        
        ### Question 9: Bond Pricing with a Given Yield
        
        A coupon bond with a par value of $1,000 and a 10-year maturity pays semiannual coupons of $21.
        
        #### (a) Price of the bond with 4% yield:
        ${:.2f}
        
        #### (b) Is the bond selling above or below par?
        {}
        
        ### Question 10: Yield to Maturity and Current Yield
        
        A coupon bond with a par value of $1,000 and a maturity of 7 years is selling for $1,040. The semiannual coupon payments are $23.
        
        #### (a) Yield to maturity:
        {:.6f} (semiannual rate) or {:.4f}% (annual rate)
        
        #### (b) Current yield:
        {:.6f} or {:.4f}%
        
        #### (c) Comparison of yields:
        {}
        
        ### Question 11: Zero-Coupon Bond Pricing with Forward Rate
        
        Given forward rate r(t) = 0.033 + 0.0012t
        
        #### Value of a par $100 zero-coupon bond with 15-year maturity:
        ${:.2f}
        
        ### Question 12: Bond Return with Changing Forward Rate
        
        Initial forward rate r(t) = 0.04 + 0.001t
        After 6 months, forward rate r(t) = 0.03 + 0.0013t
        
        #### Return on 8-year zero-coupon bond after 6 months:
        {:.6f} or {:.4f}%
        
        ### Question 13: Yield to Maturity with Piecewise Forward Rate
        
        Given forward rate r(t) = 0.03 + 0.001t - 0.0002(t-10)+
        
        #### Yield to maturity on 20-year zero-coupon bond:
        {:.6f} or {:.4f}%
        
        ## Question 14: Zero-Coupon Bond Investment Analysis
        
        ### (a) Current prices of zero-coupon bonds with par value $1,000:
        - 1-year bond: ${:.2f}
        - 3-year bond: ${:.2f}
        - 5-year bond: ${:.2f}
        
        ### (b) Prices after 1 year if spot rates remain unchanged:
        - Original 1-year bond (now matured): ${:.2f}
        - Original 3-year bond (now 2-year): ${:.2f}
        - Original 5-year bond (now 4-year): ${:.2f}
        
        ### (c) Prices after 1 year if spot rates increase by 0.005:
        - Original 1-year bond (now matured): ${:.2f}
        - Original 3-year bond (now 2-year): ${:.2f}
        - Original 5-year bond (now 4-year): ${:.2f}
        
        ### (d) Returns after 1 year if spot rates increase by 0.005:
        - 1-year bond: {:.2f}%
        - 3-year bond: {:.2f}%
        - 5-year bond: {:.2f}%
        - Best maturity if rates increase: {} bond
        
        ### (e) Returns after 1 year if spot rates remain unchanged:
        - 1-year bond: {:.2f}%
        - 3-year bond: {:.2f}%
        - 5-year bond: {:.2f}%
        - Best maturity if rates remain unchanged: {} bond
        
        ## Question 15: Bond Duration Derivative
        
        {}
        
        ## Question 16: Zero-coupon bond with yield curve
        
        ### (a) Price of 10-year zero-coupon bond:
        ${:.2f}
        
        ### (b) Price after 1 year if yield curve changes:
        ${:.2f}
        
        ### Return:
        {:.2f}%
        
        ## Question 17: Coupon bond analysis
        
        ### (a) Is the bond selling above or below par?
        {}
        
        ### (b) Is the yield to maturity above or below 2.8%?
        {}
        
        ## Question 18: Forward rate and spot rate
        
        ### (a) 5-year spot rate:
        {:.4f}%
        
        ### (b) Price of 5-year zero-coupon bond:
        ${:.2f}
        
        ## Question 19: Bond pricing with spot rates
        
        ### Price of 2-year coupon bond:
        ${:.2f}
        
        ### Cash Flows and Present Values:
        - 6 months: Cash Flow = ${:.2f}, Present Value = ${:.2f}
        - 12 months: Cash Flow = ${:.2f}, Present Value = ${:.2f}
        - 18 months: Cash Flow = ${:.2f}, Present Value = ${:.2f}
        - 24 months: Cash Flow = ${:.2f}, Present Value = ${:.2f}
        
        ## Question 20: Calculate spot rates from bond prices
        
        ### Spot Rates:
        - 0.5-year semiannual spot rate: {:.4f}%
        - 1.0-year semiannual spot rate: {:.4f}%
        - 1.5-year semiannual spot rate: {:.4f}%
        - 2.0-year semiannual spot rate: {:.4f}%
        """.format(
            # Question 1
            results['question1']['a_ytm_20yr'], results['question1']['a_ytm_20yr'] * 100,
            results['question1']['b_price_15yr'],
            
            # Question 2
            results['question2']['a_ytm_8yr'], results['question2']['a_ytm_8yr'] * 100,
            results['question2']['b_price_5yr'],
            results['question2']['d_return_1yr'], results['question2']['d_return_1yr'] * 100,
            
            # Question 3
            results['question3']['a_explanation'],
            results['question3']['b_explanation'],
            
            # Question 4
            results['question4']['a_spot_rate_5yr'], results['question4']['a_spot_rate_5yr'] * 100,
            results['question4']['b_price_5yr'],
            
            # Question 5
            results['question5']['bond_price'],
            
            # Question 6
            results['question6']['coupon'],
            results['question6']['annual_coupon'],
            results['question6']['coupon_rate'] * 100,
            
            # Question 7
            results['question7']['ytm'], results['question7']['annual_ytm'] * 100,
            
            # Question 8
            results['question8']['ytm'], results['question8']['annual_ytm'] * 100,
            
            # Question 9
            results['question9']['a_price'],
            results['question9']['b_explanation'],
            
            # Question 10
            results['question10']['a_ytm'], results['question10']['a_annual_ytm'] * 100,
            results['question10']['b_current_yield'], results['question10']['b_current_yield'] * 100,
            results['question10']['c_explanation'],
            
            # Question 11
            results['question11']['price'],
            
            # Question 12
            results['question12']['return'], results['question12']['return'] * 100,
            
            # Question 13
            results['question13']['ytm_20yr'], results['question13']['ytm_20yr'] * 100,
            
            # Question 14
            results['question14']['current_prices']['1_year'],
            results['question14']['current_prices']['3_year'],
            results['question14']['current_prices']['5_year'],
            results['question14']['prices_after_1yr_unchanged']['1_year'],
            results['question14']['prices_after_1yr_unchanged']['3_year'],
            results['question14']['prices_after_1yr_unchanged']['5_year'],
            results['question14']['prices_after_1yr_increased']['1_year'],
            results['question14']['prices_after_1yr_increased']['3_year'],
            results['question14']['prices_after_1yr_increased']['5_year'],
            results['question14']['returns_increased']['1_year'],
            results['question14']['returns_increased']['3_year'],
            results['question14']['returns_increased']['5_year'],
            results['question14']['best_maturity_increased'],
            results['question14']['returns_unchanged']['1_year'],
            results['question14']['returns_unchanged']['3_year'],
            results['question14']['returns_unchanged']['5_year'],
            results['question14']['best_maturity_unchanged'],
            
            # Question 15
            results['question15'],
            
            # Question 16
            results['question16']['a_price'],
            results['question16']['b_price_after_1yr'],
            results['question16']['b_return'] * 100,
            
            # Question 17
            results['question17']['a_explanation'],
            results['question17']['b_explanation'],
            
            # Question 18
            results['question18']['a_spot_rate'] * 100,
            results['question18']['b_price'],
            
            # Question 19
            results['question19']['bond_price'],
            results['question19']['cash_flows']['6_months']['cash_flow'],
            results['question19']['cash_flows']['6_months']['present_value'],
            results['question19']['cash_flows']['12_months']['cash_flow'],
            results['question19']['cash_flows']['12_months']['present_value'],
            results['question19']['cash_flows']['18_months']['cash_flow'],
            results['question19']['cash_flows']['18_months']['present_value'],
            results['question19']['cash_flows']['24_months']['cash_flow'],
            results['question19']['cash_flows']['24_months']['present_value'],
            
            # Question 20
            results['question20']['spot_rates'][0.5],
            results['question20']['spot_rates'][1.0],
            results['question20']['spot_rates'][1.5],
            results['question20']['spot_rates'][2.0]
        )
        
        report_path = os.path.join(self.output_dir, 'bond_calculator_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path

def main():
    parser = argparse.ArgumentParser(description='Bond Calculator')
    parser.add_argument('--question', type=int, choices=range(1, 21), help='Specific question to solve (1-20)')
    parser.add_argument('--all', action='store_true', help='Run all calculations')
    parser.add_argument('--report', action='store_true', help='Generate a comprehensive report')
    
    args = parser.parse_args()
    
    calculator = BondCalculator()
    
    if args.all or args.report:
        results = calculator.run_all_calculations()
        
        if args.report:
            report_path = calculator.generate_report(results)
            print(f"Report generated at: {report_path}")
        else:
            print("All calculations completed successfully.")
            print(f"Results and plots saved in: {calculator.output_dir}")
    
    elif args.question:
        if args.question == 1:
            result = calculator.solve_question1()
            print("Question 1: Forward Rate and Bond Pricing")
            print(f"(a) Yield to maturity for T=20 years = {result['a_ytm_20yr']:.6f} or {result['a_ytm_20yr']*100:.4f}%")
            print(f"(b) Price of zero-coupon bond maturing in 15 years = ${result['b_price_15yr']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 2:
            result = calculator.solve_question2()
            print("Question 2: Forward Rate, Yield Curve, and Bond Returns")
            print(f"(a) Yield to maturity for T=8 years = {result['a_ytm_8yr']:.6f} or {result['a_ytm_8yr']*100:.4f}%")
            print(f"(b) Price of zero-coupon bond maturing in 5 years = ${result['b_price_5yr']:.2f}")
            print(f"(d) Return after 1 year = {result['d_return_1yr']:.6f} or {result['d_return_1yr']*100:.4f}%")
            print(f"Plot saved at: {result['c_plot_path']}")
        
        elif args.question == 3:
            result = calculator.solve_question3()
            print("Question 3: Coupon Bond Analysis")
            print(f"(a) {result['a_explanation']}")
            print(f"(b) {result['b_explanation']}")
        
        elif args.question == 4:
            result = calculator.solve_question4()
            print("Question 4: Forward Rate and Spot Rate")
            print(f"(a) 5-year continuously compounded spot rate = {result['a_spot_rate_5yr']:.6f} or {result['a_spot_rate_5yr']*100:.4f}%")
            print(f"(b) Price of zero-coupon bond maturing in 5 years = ${result['b_price_5yr']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 5:
            result = calculator.solve_question5()
            print("Question 5: Bond Pricing with Spot Rates")
            print(f"Price of 2-year coupon bond with semiannual payments = ${result['bond_price']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 6:
            result = calculator.solve_question6()
            print("Question 6: Finding Coupon Payment Given Yield to Maturity")
            print(f"Coupon payment = ${result['coupon']:.2f} (semiannual)")
            print(f"Annual coupon payment = ${result['annual_coupon']:.2f}")
            print(f"Coupon rate = {result['coupon_rate']*100:.4f}%")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 7:
            result = calculator.solve_question7()
            print("Question 7: Yield to Maturity of a 30-year Bond")
            print(f"Yield to maturity = {result['ytm']:.6f} (semiannual rate) or {result['annual_ytm']*100:.4f}% (annual rate)")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 8:
            result = calculator.solve_question8()
            print("Question 8: Yield to Maturity of a 20-year Bond")
            print(f"Yield to maturity = {result['ytm']:.6f} (semiannual rate) or {result['annual_ytm']*100:.4f}% (annual rate)")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 9:
            result = calculator.solve_question9()
            print("Question 9: Bond Pricing with a Given Yield")
            print(f"(a) Price of bond with 4% yield = ${result['a_price']:.2f}")
            print(f"(b) {result['b_explanation']}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 10:
            result = calculator.solve_question10()
            print("Question 10: Yield to Maturity and Current Yield")
            print(f"(a) Yield to maturity = {result['a_ytm']:.6f} (semiannual rate) or {result['a_annual_ytm']*100:.4f}% (annual rate)")
            print(f"(b) Current yield = {result['b_current_yield']:.6f} or {result['b_current_yield']*100:.4f}%")
            print(f"(c) {result['c_explanation']}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 11:
            result = calculator.solve_question11()
            print("Question 11: Zero-Coupon Bond Pricing with Forward Rate")
            print(f"Value of $100 zero-coupon bond with 15-year maturity = ${result['price']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 12:
            result = calculator.solve_question12()
            print("Question 12: Bond Return with Changing Forward Rate")
            print(f"Return after 6 months = {result['return']:.6f} or {result['return']*100:.4f}%")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 13:
            result = calculator.solve_question13()
            print("Question 13: Yield to Maturity with Piecewise Forward Rate")
            print(f"Yield to maturity on 20-year zero-coupon bond = {result['ytm_20yr']:.6f} or {result['ytm_20yr']*100:.4f}%")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 14:
            result = calculator.zero_coupon_bond_investment_analysis()
            print("Question 14: Zero-Coupon Bond Investment Analysis")
            print(f"Best maturity if rates increase: {result['best_maturity_increased']} bond")
            print(f"Best maturity if rates remain unchanged: {result['best_maturity_unchanged']} bond")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 15:
            result = calculator.bond_duration_derivative()
            print("Question 15: Bond Duration Derivative")
            print(result)
        
        elif args.question == 16:
            result = calculator.zero_coupon_bond_with_yield_curve()
            print("Question 16: Zero-coupon bond with yield curve")
            print(f"(a) Price of 10-year zero-coupon bond: ${result['a_price']:.2f}")
            print(f"(b) Price after 1 year: ${result['b_price_after_1yr']:.2f}")
            print(f"    Return: {result['b_return']*100:.2f}%")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 17:
            result = calculator.coupon_bond_analysis()
            print("Question 17: Coupon bond analysis")
            print(f"(a) {result['a_explanation']}")
            print(f"(b) {result['b_explanation']}")
        
        elif args.question == 18:
            result = calculator.forward_rate_and_spot_rate()
            print("Question 18: Forward rate and spot rate")
            print(f"(a) 5-year spot rate: {result['a_spot_rate']*100:.4f}%")
            print(f"(b) Price of 5-year zero-coupon bond: ${result['b_price']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 19:
            result = calculator.bond_pricing_with_spot_rates()
            print("Question 19: Bond pricing with spot rates")
            print(f"Price of 2-year coupon bond: ${result['bond_price']:.2f}")
            print(f"Plot saved at: {result['plot_path']}")
        
        elif args.question == 20:
            result = calculator.calculate_spot_rates_from_prices()
            print("Question 20: Calculate spot rates from bond prices")
            for t, rate in result['spot_rates'].items():
                print(f"{t}-year semiannual spot rate: {rate:.4f}%")
            print(f"Plot saved at: {result['plot_path']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
