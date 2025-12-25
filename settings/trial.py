import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# --- PATHS FROM YOUR SCREENSHOTS ---
lstm_folder = r"C:/Users/dgajjar7/Desktop/trading/results/experiment_quandl_100assets_lstm_cpnone_len63_notime_div_v1"
tft_folder  = r"C:/Users/dgajjar7/Desktop/trading/results/experiment_quandl_100assets_tft_cpnone_len252_notime_div_v1"

def load_fw_returns(experiment_folder):
    all_data = []
    # Find all year folders (e.g., 2016-2017)
    year_folders = sorted([f for f in glob.glob(os.path.join(experiment_folder, "20*")) if os.path.isdir(f)])
    
    print(f"Reading {len(year_folders)} years from: {os.path.basename(experiment_folder)}")
    
    for folder in year_folders:
        # Construct path to the results CSV
        target_file = os.path.join(folder, "best", "captured_returns_fw.csv")
        
        if os.path.exists(target_file):
            try:
                df = pd.read_csv(target_file)
                # Ensure date column is datetime
                if 'time' in df.columns and 'captured_returns' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    # Average across assets to get daily portfolio return
                    daily_mean = df.groupby(level=0)['captured_returns'].mean()
                    all_data.append(daily_mean)
            except:
                continue
    
    if all_data:
        return pd.concat(all_data).sort_index()
    else:
        return pd.Series()

# --- MAIN EXECUTION ---
print("Loading data...")
lstm_returns = load_fw_returns(lstm_folder)
tft_returns = load_fw_returns(tft_folder)

if lstm_returns.empty or tft_returns.empty:
    print("ERROR: No data found. Check your paths.")
else:
    # Calculate Cumulative Growth (Equity Curve)
    lstm_cum = (1 + lstm_returns).cumprod()
    tft_cum = (1 + tft_returns).cumprod()

    # Align Data
    combined = pd.DataFrame({
        'LSTM (Benchmark)': lstm_cum,
        'Momentum Transformer': tft_cum
    }).dropna()

    # --- CALCULATE SHARPE RATIOS FOR SLIDE 8 ---
    # Sharpe = mean / std * sqrt(252)
    sharpe_lstm = (lstm_returns.mean() / lstm_returns.std()) * np.sqrt(252)
    sharpe_tft = (tft_returns.mean() / tft_returns.std()) * np.sqrt(252)

    print("\n" + "="*30)
    print("RESULTS FOR SLIDE 8")
    print("="*30)
    print(f"LSTM Sharpe Ratio: {sharpe_lstm:.2f}")
    print(f"Transformer Sharpe Ratio: {sharpe_tft:.2f}")
    print("="*30 + "\n")

    # --- PLOT FOR SLIDE 6 ---
    plt.figure(figsize=(12, 7))
    plt.plot(combined.index, combined['LSTM (Benchmark)'], color='grey', linestyle='--', label=f'LSTM (Sharpe: {sharpe_lstm:.2f})')
    plt.plot(combined.index, combined['Momentum Transformer'], color='#004488', linewidth=2.5, label=f'Transformer (Sharpe: {sharpe_tft:.2f})')
    
    plt.title('Cumulative Returns: Transformer vs LSTM', fontsize=14)
    plt.ylabel('Growth of $1 Investment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight Covid Crash
    plt.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), color='red', alpha=0.1)
    plt.text(pd.Timestamp('2020-03-01'), combined['Momentum Transformer'].max(), 'Covid Crisis', color='red')

    plt.tight_layout()
    plt.savefig('combined_returns.png') # Saves the file
    plt.show()
    print("Chart saved as 'combined_returns.png'. Put this in Slide 6!")