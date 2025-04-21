#!/usr/bin/env python3
from bond_calculator_cli import BondCalculator
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create an instance of the BondCalculator
    bond_calc = BondCalculator()
    
    print("===================================================")
    print("         BOND CALCULATOR SAMPLE CALCULATIONS        ")
    print("===================================================")
    
    # Example 1: Calculate bond price
    print("\n--- Example 1: Calculate Bond Price ---")
    
    # Parameters
    par = 1000
    coupon_rate = 0.05  # 5% annual coupon rate
    maturity = 10  # 10 years
    ytm = 0.04  # 4% yield to maturity
    
    # Convert to semi-annual
    semiannual_coupon = coupon_rate * par / 2
    semiannual_ytm = ytm / 2
    
    # Calculate bond price
    price = bond_calc.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
    
    # Print results
    print(f"Parameters:")
    print(f"  Par Value: ${par}")
    print(f"  Annual Coupon Rate: {coupon_rate*100:.2f}%")
    print(f"  Time to Maturity: {maturity} years")
    print(f"  Annual Yield to Maturity: {ytm*100:.2f}%")
    print(f"\nResults:")
    print(f"  Bond Price: ${price:.2f}")
    print(f"  The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.")
    
    # Example 2: Find yield to maturity
    print("\n--- Example 2: Find Yield to Maturity ---")
    
    # Parameters
    par = 1000
    price = 950
    coupon_rate = 0.05  # 5% annual coupon rate
    maturity = 10  # 10 years
    
    # Convert to semi-annual
    semiannual_coupon = coupon_rate * par / 2
    
    # Calculate YTM
    semiannual_ytm = bond_calc.find_ytm_using_root(price, semiannual_coupon, maturity, par)
    annual_ytm = semiannual_ytm * 2
    
    # Calculate current yield
    current_yield = (coupon_rate * par) / price
    
    # Print results
    print(f"Parameters:")
    print(f"  Bond Price: ${price}")
    print(f"  Par Value: ${par}")
    print(f"  Annual Coupon Rate: {coupon_rate*100:.2f}%")
    print(f"  Time to Maturity: {maturity} years")
    print(f"\nResults:")
    print(f"  Yield to Maturity: {annual_ytm*100:.4f}%")
    print(f"  Current Yield: {current_yield*100:.4f}%")
    print(f"  The bond is trading at a {'premium' if price > par else 'discount' if price < par else 'par'}.")
    
    # Example 3: Calculate bond duration
    print("\n--- Example 3: Calculate Bond Duration ---")
    
    # Parameters
    par = 1000
    coupon_rate = 0.05  # 5% annual coupon rate
    maturity = 10  # 10 years
    ytm = 0.04  # 4% yield to maturity
    
    # Convert to semi-annual
    semiannual_coupon = coupon_rate * par / 2
    semiannual_ytm = ytm / 2
    
    # Calculate duration
    duration = bond_calc.calculate_bond_duration(semiannual_coupon, maturity, semiannual_ytm, par)
    
    # Calculate modified duration
    modified_duration = duration / (1 + semiannual_ytm)
    
    # Calculate price for additional info
    price = bond_calc.bondvalue(semiannual_coupon, maturity, semiannual_ytm, par)
    
    # Print results
    print(f"Parameters:")
    print(f"  Par Value: ${par}")
    print(f"  Annual Coupon Rate: {coupon_rate*100:.2f}%")
    print(f"  Time to Maturity: {maturity} years")
    print(f"  Annual Yield to Maturity: {ytm*100:.2f}%")
    print(f"\nResults:")
    print(f"  Bond Price: ${price:.2f}")
    print(f"  Macaulay Duration: {duration:.4f} years")
    print(f"  Modified Duration: {modified_duration:.4f}")
    print(f"  For a 1% increase in yield, the bond price would decrease by approximately {modified_duration*100:.2f}%")
    print(f"  For a 1% decrease in yield, the bond price would increase by approximately {modified_duration*100:.2f}%")
    
    # Example 4: Create a price vs. yield curve
    print("\n--- Example 4: Price vs. Yield Curve ---")
    print("Generating price vs. yield curve...")
    
    # Parameters
    par = 1000
    coupon_rate = 0.05  # 5% annual coupon rate
    maturity = 10  # 10 years
    
    # Create array of yields
    yields = np.linspace(0.01, 0.10, 100)  # 1% to 10%
    
    # Convert to semi-annual
    semiannual_coupon = coupon_rate * par / 2
    semiannual_yields = yields / 2
    
    # Calculate bond prices for each yield
    prices = [bond_calc.bondvalue(semiannual_coupon, maturity, y, par) for y in semiannual_yields]
    
    # Plot the price vs. yield curve
    plt.figure(figsize=(10, 6))
    plt.plot(yields * 100, prices, 'b-')
    plt.title(f'Bond Price vs. Yield Curve\n(Coupon: {coupon_rate*100}%, Maturity: {maturity} years)')
    plt.xlabel('Yield to Maturity (%)')
    plt.ylabel('Bond Price ($)')
    plt.grid(True)
    
    # Save the plot
    plt.savefig('price_vs_yield_curve.png')
    print("Price vs. yield curve saved as 'price_vs_yield_curve.png'")
    
    # Example 5: Duration comparison for different coupon rates
    print("\n--- Example 5: Duration for Different Coupon Rates ---")
    print("Calculating durations for different coupon rates...")
    
    # Parameters
    par = 1000
    ytm = 0.05  # 5% yield to maturity
    maturity = 20  # 20 years
    
    # Array of coupon rates
    coupon_rates = np.linspace(0.02, 0.08, 7)  # 2% to 8%
    
    # Convert to semi-annual
    semiannual_ytm = ytm / 2
    
    # Calculate durations for each coupon rate
    durations = []
    for rate in coupon_rates:
        semiannual_coupon = rate * par / 2
        duration = bond_calc.calculate_bond_duration(semiannual_coupon, maturity, semiannual_ytm, par)
        durations.append(duration)
    
    # Plot the durations
    plt.figure(figsize=(10, 6))
    plt.plot(coupon_rates * 100, durations, 'r-o')
    plt.title(f'Bond Duration vs. Coupon Rate\n(YTM: {ytm*100}%, Maturity: {maturity} years)')
    plt.xlabel('Coupon Rate (%)')
    plt.ylabel('Duration (years)')
    plt.grid(True)
    
    # Save the plot
    plt.savefig('duration_vs_coupon_rate.png')
    print("Duration vs. coupon rate curve saved as 'duration_vs_coupon_rate.png'")
    
    print("\nAll examples completed successfully.")

if __name__ == "__main__":
    main()