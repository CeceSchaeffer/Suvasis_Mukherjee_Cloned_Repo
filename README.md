# Bond Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bond-analytics.streamlit.app/)

A comprehensive bond calculator and analytics platform built with Streamlit. This application provides tools for bond pricing, yield analysis, duration calculations, and real financial data visualization.

## Features

### Basic Bond Calculator
- Calculate bond prices based on par value, coupon rate, and yield to maturity
- Compute current yield, Macaulay duration, and modified duration
- Visualize cash flows and price-yield relationships
- Analyze how changes in yield affect bond prices

### Advanced Bond Functions
- Forward rate and spot rate calculations with custom rate functions
- Bond pricing using different spot rates for each cash flow
- Finding yield to maturity (YTM) using numerical methods
- Determining required coupon rates for specific pricing scenarios

### Real Data Analysis
- Interactive yield curve visualization from actual market data
- Treasury yield curve analysis across different time periods
- Zero-coupon bond price and yield analysis
- Visualization of the relationship between spot rates and forward rates

### Educational Resources
- Detailed explanations of key bond concepts
- Interactive learning materials on:
  - Yield to Maturity (YTM)
  - Bond Duration
  - Forward Rates
  - Yield Curves
  - Zero-Coupon Bonds

### Interactive Chat Interface
- Ask questions about bond calculations and concepts
- Get answers with formulas and explanations
- Learn bond market fundamentals through conversation

## Getting Started

### Online Access
The application is deployed at [bondanalytics.streamlit.app](https://bondanalytics.streamlit.app) - visit this link to use the app without installation.

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/dronomyio/bond_analytics.git
cd bond_analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py  or
streamlit run app.py  --server.port 8509 #or any other port
```

### Docker Deployment

```bash
docker build -t bond-analytics .
docker run -p 8505:8505 bond-analytics

Example:
docker run -p 8508:8508 bond-analytics --question 4

usage: bond_calculator.py [-h]
                          [--question {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}]
                          [--all] [--report]

Bond Calculator

options:
  -h, --help            show this help message and exit
  --question {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
                        Specific question to solve (1-20)
  --all                 Run all calculations
  --report              Generate a comprehensive report
```

## Data Sources
The application includes datasets for:
- Historical yield curves
- Treasury yield data
- Zero-coupon bond prices
- Stock and bond market data

All data is stored in the `data/datasets` directory.

## Key Bond Formulas

### Bond Price
For a bond with semiannual payments:
```
P = C/r + (Par - C/r) * (1 + r)^(-2T)
```
Where:
- P = Bond price
- C = Semiannual coupon payment
- r = Semiannual yield to maturity
- T = Time to maturity in years
- Par = Par value of the bond

### Macaulay Duration
```
Duration = Σ(t × PV(CFt)) / Price
```
Where:
- t = Time to each cash flow
- PV(CFt) = Present value of the cash flow at time t
- Price = Bond price

### Modified Duration
```
Modified Duration = Macaulay Duration / (1 + YTM/n)
```
Where n is the number of coupon payments per year.

### Forward Rate
```
Forward Rate(t₁,t₂) = (Spot Rate(t₂) × t₂ - Spot Rate(t₁) × t₁) / (t₂ - t₁)
```

## Contributing
Contributions to enhance the application are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License
This project is available under the dronomy.io License.
