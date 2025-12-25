# Momentum Transformer: Modern Implementation (TF 2.x) 🚀

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-blue)
![Status](https://img.shields.io/badge/Status-Refactored%20%26%20Optimized-green)

## 📄 Project Overview
This repository contains a **re-engineered implementation** of the *Momentum Transformer* (Wood et al., 2021). 

While the original research relied on paid institutional data (Quandl) and legacy dependencies, this implementation adapts the architecture to be **accessible and reproducible** on consumer hardware.

## 🛠️ Key Engineering Adaptations
* [cite_start]**Data Pipeline Migration:** Replaced the deprecated/paid Quandl data feed with a robust scraper for **Yahoo Finance**, engineering continuous futures data from ~100 liquid tickers[cite: 197].
* **Framework Upgrade:** Ported the model architecture to **TensorFlow 2.x** (from legacy versions), utilizing modern Keras functional APIs for the Attention mechanisms.
* **Local Optimization:** Refactored the training loop to run efficiently on a single GPU/CPU setup (Personal PC) rather than requiring a research cluster.

## 📊 Results (1990-2025 Backtest)
Despite the data source change, the model successfully reproduced the "Crisis Alpha" behavior described in the original paper.

| Metric | LSTM Baseline | **Momentum Transformer (My Implementation)** |
| :--- | :--- | :--- |
| **Total Return** | 113.90% | **1248.99%** |
| **Sharpe Ratio** | 1.23 | **2.06** |
| **Max Drawdown** | -7.74% | **-7.19%** |

*(See `rEport.pdf` for full performance breakdown)*

## 🧠 Architecture
The model utilizes a **Decoder-Only Temporal Fusion Transformer (TFT)** with:
* **Multi-Head Attention:** To capture long-term dependencies across decades.
* **Variable Selection Networks:** To weigh the importance of daily vs. monthly returns dynamically.

## 🚀 Usage
1.  **Clone:** `git clone https://github.com/DixitMGajjar/Trading-with-the-Momentum-Transformer`
2.  **Install:** `pip install -r requirements.txt`
3.  **Run Pipeline:**
    * `python data_loader.py` (Fetches Yahoo Finance data)
    * `python train_tft.py` (Trains the TensorFlow model)
