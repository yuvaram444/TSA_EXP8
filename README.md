# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 25-05-26


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
"""
AIM:
To implement Moving Average Model and Exponential Smoothing using Python
on the Nifty 50 stock dataset (HDFCBANK.NS) to smooth the monthly average
close price time series, analyze rolling window transformations, and apply
exponential smoothing to capture underlying trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# ── 1. Import & Read CSV ──────────────────────────────────────────────────────
STOCK    = 'HDFCBANK.NS'
CSV_PATH = 'nifty50_2000_2025.csv'   # keep CSV in same folder as notebook

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

stock_df = df[df['Stock'] == STOCK].sort_values('Date').copy()
stock_df['YearMonth'] = stock_df['Date'].dt.to_period('M')

monthly = stock_df.groupby('YearMonth')['Close'].mean()
monthly.index = monthly.index.to_timestamp()
monthly.index.freq = 'MS'

# ── 2. Display shape and first 20 rows ───────────────────────────────────────
print("=" * 55)
print("DATASET INFO")
print("=" * 55)
print(f"\nStock       : {STOCK}")
print(f"Shape       : {monthly.shape}")
print(f"Date range  : {monthly.index[0].date()} → {monthly.index[-1].date()}")
print(f"\nFirst 20 rows:")
print(monthly.head(20).to_frame())

# ── 3. Set figure size & 4. Suppress warnings ─────────────────────────────────
plt.rcParams['figure.figsize'] = [14, 5]
warnings.filterwarnings('ignore')

# ── 5. Plot first 50 values ───────────────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.plot(monthly.iloc[:50], color='steelblue', linewidth=1.5, marker='o', markersize=3)
plt.title(f'{STOCK} — First 50 Monthly Avg Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 6. Rolling Average — Window size 5 ───────────────────────────────────────
rolling_mean_5 = monthly.rolling(window=5).mean()

# ── 7. Display first 10 values of rolling mean (window=5) ────────────────────
print("\n" + "=" * 55)
print("MOVING AVERAGE")
print("=" * 55)
print("\nRolling Mean (window=5) — First 10 values:")
print(rolling_mean_5.head(10).to_frame())

# ── 8. Rolling Average — Window size 10 ──────────────────────────────────────
rolling_mean_10 = monthly.rolling(window=10).mean()

print("\nRolling Mean (window=10) — First 10 values:")
print(rolling_mean_10.head(10).to_frame())

# ── 9 & 10. Plot original + both rolling means ────────────────────────────────
print("\n" + "=" * 55)
print("PLOT TRANSFORM DATASET")
print("=" * 55)

plt.figure(figsize=(14, 5))
plt.plot(monthly,         color='steelblue',  linewidth=1.2, alpha=0.6, label='Original Monthly Close')
plt.plot(rolling_mean_5,  color='tomato',     linewidth=2,               label='Rolling Mean (window=5)')
plt.plot(rolling_mean_10, color='darkorange', linewidth=2,               label='Rolling Mean (window=10)')
plt.title(f'{STOCK} — Moving Average Transformation (Window 5 & 10)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Rolling mean on first 50 values (zoomed view — like original code)
plt.figure(figsize=(14, 5))
plt.plot(monthly.iloc[:50],          color='steelblue',  linewidth=1.5, alpha=0.7, label='Original (first 50)')
plt.plot(rolling_mean_5.iloc[:50],   color='tomato',     linewidth=2,               label='Rolling Mean (window=5)')
plt.plot(rolling_mean_10.iloc[:50],  color='darkorange', linewidth=2,               label='Rolling Mean (window=10)')
plt.title(f'{STOCK} — Moving Average (First 50 Months, Zoomed)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 11. Exponential Smoothing ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("EXPONENTIAL SMOOTHING")
print("=" * 55)

# Fit with three alpha values to show effect of smoothing factor
alphas = [0.1, 0.3, 0.6]
colors = ['tomato', 'darkorange', 'green']

plt.figure(figsize=(14, 5))
plt.plot(monthly, color='steelblue', linewidth=1.2, alpha=0.5, label='Original Monthly Close')

for alpha, color in zip(alphas, colors):
    model  = SimpleExpSmoothing(monthly).fit(smoothing_level=alpha, optimized=False)
    fitted = model.fittedvalues
    plt.plot(fitted, color=color, linewidth=1.8, label=f'Exp Smoothing (α={alpha})')
    print(f"\nα = {alpha}  →  First 5 smoothed values:")
    print(fitted.head().round(2).to_frame(name=f'Smoothed (α={alpha})'))

plt.title(f'{STOCK} — Exponential Smoothing (α = 0.1, 0.3, 0.6)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Zoomed — first 50 months
plt.figure(figsize=(14, 5))
plt.plot(monthly.iloc[:50], color='steelblue', linewidth=1.5, alpha=0.6, label='Original (first 50)')
for alpha, color in zip(alphas, colors):
    model  = SimpleExpSmoothing(monthly).fit(smoothing_level=alpha, optimized=False)
    fitted = model.fittedvalues
    plt.plot(fitted.iloc[:50], color=color, linewidth=1.8, label=f'Exp Smoothing (α={alpha})')
plt.title(f'{STOCK} — Exponential Smoothing (First 50 Months, Zoomed)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRESULT:")
print("Thus the program ran successfully based on Moving Average and Exponential Smoothing.")
```

### OUTPUT:
```
=======================================================
DATASET INFO
=======================================================

Stock       : HDFCBANK.NS
Shape       : (299,)
Date range  : 2000-02-01 → 2024-12-01

First 20 rows:
                Close
YearMonth            
2000-02-01  11.551923
2000-03-01  13.150000
2000-04-01  11.367125
2000-05-01  12.426304
2000-06-01  12.466023
2000-07-01  12.882143
2000-08-01  11.968152
2000-09-01  11.541071
2000-10-01  12.014205
2000-11-01  11.744091
2000-12-01  10.994048
2001-01-01  12.017935
2001-02-01  12.809000
2001-03-01  12.069091
2001-04-01  11.670833
2001-05-01  11.740761
2001-06-01  11.021309
2001-07-01  10.928977
2001-08-01  11.695326
2001-09-01  10.947875
```
<img width="1116" height="401" alt="image" src="https://github.com/user-attachments/assets/453b1991-4680-4964-88a8-20170875e640" />


Moving Average
```
=======================================================
MOVING AVERAGE
=======================================================

Rolling Mean (window=5) — First 10 values:
                Close
YearMonth            
2000-02-01        NaN
2000-03-01        NaN
2000-04-01        NaN
2000-05-01        NaN
2000-06-01  12.192275
2000-07-01  12.458319
2000-08-01  12.221949
2000-09-01  12.256739
2000-10-01  12.174319
2000-11-01  12.029932

Rolling Mean (window=10) — First 10 values:
                Close
YearMonth            
2000-02-01        NaN
2000-03-01        NaN
2000-04-01        NaN
2000-05-01        NaN
2000-06-01        NaN
2000-07-01        NaN
2000-08-01        NaN
2000-09-01        NaN
2000-10-01        NaN
2000-11-01  12.111104
```

Plot Transform Dataset
<img width="1118" height="803" alt="image" src="https://github.com/user-attachments/assets/216b90c3-9e07-41e1-94ab-2e699962c048" />



Exponential Smoothing
```
α = 0.1  →  First 5 smoothed values:
            Smoothed (α=0.1)
YearMonth                   
2000-02-01             11.55
2000-03-01             11.55
2000-04-01             11.71
2000-05-01             11.68
2000-06-01             11.75

α = 0.3  →  First 5 smoothed values:
            Smoothed (α=0.3)
YearMonth                   
2000-02-01             11.55
2000-03-01             11.55
2000-04-01             12.03
2000-05-01             11.83
2000-06-01             12.01

α = 0.6  →  First 5 smoothed values:
            Smoothed (α=0.6)
YearMonth                   
2000-02-01             11.55
2000-03-01             11.55
2000-04-01             12.51
2000-05-01             11.82
2000-06-01             12.19
```
<img width="1120" height="812" alt="image" src="https://github.com/user-attachments/assets/e49c35fa-d00f-4d30-8d80-34c885dc60a6" />




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
