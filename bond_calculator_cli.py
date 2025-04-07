#!/usr/bin/env python3
import numpy as np
from scipy import integrate
from scipy.optimize import brentq
import os

class BondCalculator:
    """
    A simplified bond calculator for command-line usage
    """
    
    def __init__(self):
        """Initialize the BondCalculator class"""
        pass
    
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
    
    def find_ytm_using_root(self, price, c, T, par, r_min=0.001, r_max=0.2):
        """Find yield to maturity using root-finding method"""
        def bond_price_diff(r, price, c, T, par):
            calculated_price = self.bondvalue(c, T, r, par)
            return calculated_price - price
        
        ytm = brentq(bond_price_diff, r_min, r_max, args=(price, c, T, par))
        return ytm
    
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


def main():
    """Main function to run the bond calculator CLI"""
    bond_calc = BondCalculator()
    
    print("=======================================")
    print("         BOND CALCULATOR CLI           ")
    print("=======================================\n")
    
    while True:
        print("\nChoose an option:")
        print("1. Calculate bond value")
        print("2. Find yield to maturity")
        print("3. Calculate duration")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Calculate bond value
            print("\n--- Calculate Bond Value ---")
            try:
                coupon_rate = float(input("Enter annual coupon rate (as a decimal, e.g., 0.05 for 5%): "))
                maturity = float(input("Enter time to maturity (in years): "))
                ytm = float(input("Enter annual yield to maturity (as a decimal, e.g., 0.04 for 4%): "))
                par = float(input("Enter par value (default 1000): ") or "1000")
                
                # Convert to semi-annual
                semiannual_coupon = coupon_rate * par / 2
                semiannual_ytm = ytm / 2
                
                price = bond_calc.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
                
                print("\nResults:")
                print(f"Bond Value: ${price:.2f}")
                print(f"The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.")
                
            except ValueError as e:
                print(f"Error: {e}. Please enter valid numbers.")
                
        elif choice == '2':
            # Find yield to maturity
            print("\n--- Find Yield to Maturity ---")
            try:
                price = float(input("Enter bond price: "))
                coupon_rate = float(input("Enter annual coupon rate (as a decimal, e.g., 0.05 for 5%): "))
                maturity = float(input("Enter time to maturity (in years): "))
                par = float(input("Enter par value (default 1000): ") or "1000")
                
                # Convert to semi-annual
                semiannual_coupon = coupon_rate * par / 2
                
                # Calculate YTM
                semiannual_ytm = bond_calc.find_ytm_using_root(price, semiannual_coupon, maturity, par)
                annual_ytm = semiannual_ytm * 2
                
                # Calculate current yield
                current_yield = (coupon_rate * par) / price
                
                print("\nResults:")
                print(f"Yield to Maturity: {annual_ytm*100:.4f}%")
                print(f"Current Yield: {current_yield*100:.4f}%")
                print(f"The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.")
                
            except ValueError as e:
                print(f"Error: {e}. Please enter valid numbers.")
                
        elif choice == '3':
            # Calculate duration
            print("\n--- Calculate Bond Duration ---")
            try:
                coupon_rate = float(input("Enter annual coupon rate (as a decimal, e.g., 0.05 for 5%): "))
                maturity = float(input("Enter time to maturity (in years): "))
                ytm = float(input("Enter annual yield to maturity (as a decimal, e.g., 0.04 for 4%): "))
                par = float(input("Enter par value (default 1000): ") or "1000")
                
                # Convert to semi-annual
                semiannual_coupon = coupon_rate * par / 2
                semiannual_ytm = ytm / 2
                
                # Calculate duration
                duration = bond_calc.calculate_bond_duration(semiannual_coupon, maturity, semiannual_ytm, par)
                
                # Calculate price for additional info
                price = bond_calc.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
                
                print("\nResults:")
                print(f"Bond Duration: {duration:.4f} years")
                print(f"Bond Value: ${price:.2f}")
                print(f"Duration represents the weighted average time to receive the bond's cash flows.")
                print(f"It also indicates the approximate percentage change in price for a 1% change in yield.")
                
            except ValueError as e:
                print(f"Error: {e}. Please enter valid numbers.")
                
        elif choice == '4':
            # Exit
            print("\nExiting Bond Calculator. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please select a number between 1 and 4.")


if __name__ == "__main__":
    main()