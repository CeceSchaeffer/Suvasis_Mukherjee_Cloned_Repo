# Bond Calculator

A comprehensive bond calculator with multiple interfaces for financial calculations.

## Features

- Calculate bond price given coupon rate, maturity, and yield to maturity
- Find yield to maturity given bond price, coupon rate, and maturity
- Calculate bond duration and modified duration
- Generate price vs. yield curves
- Analyze the relationship between coupon rates and duration
- Access to real financial data for yield curves, treasury data, and stock correlations
- Upload custom datasets

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd enhanced_bond_calculator
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage Options

### Chat Interface

Run the interactive chat-based interface:

```
python bond_chat_cli.py
```

This provides a conversation-style interface where you can:
- Ask questions about bond calculations
- Run specific calculations with parameters
- View real financial data
- Generate visualizations
- Upload custom datasets

Example commands in the chat interface:
- `calculate bond coupon 5% maturity 10 ytm 4%`
- `find ytm price 950 coupon 5% maturity 10`
- `show yield curve`
- `help`

### Command-Line Calculator

Run the simple command-line calculator:

```
python bond_calculator_cli.py
```

This will display a menu with the following options:
1. Calculate bond value
2. Find yield to maturity
3. Calculate duration
4. Exit

### Sample Calculations

To see examples of various bond calculations, run:

```
python sample_bond_calculations.py
```

This script demonstrates:
- Bond price calculation
- Yield to maturity calculation
- Duration calculation
- Price vs. yield curve generation
- Duration vs. coupon rate analysis

### Web Interface (requires Streamlit)

For a full web-based UI with advanced features:

```
streamlit run app.py
```

## Data Management

The application can use financial datasets from the `data/datasets` directory. You can:

1. Use the included sample datasets
2. Upload custom datasets through the chat interface
3. Add your own files directly to the data directory

## Files

- `bond_chat_cli.py`: Interactive chat interface for bond calculations
- `bond_calculator_cli.py`: Simple command-line interface
- `sample_bond_calculations.py`: Script with examples of bond calculations
- `app.py`: Streamlit web application for full visual interface
- `requirements.txt`: Dependencies for the project
- `data/`: Directory containing financial datasets

## Dockerfile

A Dockerfile is included to containerize the application if needed:

```
docker build -t bond-calculator .
docker run -p 8505:8505 bond-calculator
```

## Bond Formulas Used

- **Bond Price**: `P = C/r + (Par - C/r) * (1 + r)^(-2T)`
- **Duration**: The weighted average time to receive the bond's cash flows
- **Modified Duration**: `MD = Duration / (1 + r)`
- **Yield to Maturity**: The interest rate that makes the present value of future cash flows equal to the current price
- **Forward Rate**: `F(t₁,t₂) = (R(t₂)·t₂ - R(t₁)·t₁) / (t₂ - t₁)`

Where:
- `P` = Bond price
- `C` = Semiannual coupon payment
- `r` = Semiannual yield to maturity
- `T` = Time to maturity in years
- `Par` = Par value of the bond
- `R(t)` = Spot rate for maturity t
- `F(t₁,t₂)` = Forward rate between times t₁ and t₂