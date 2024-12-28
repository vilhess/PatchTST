# Implementation of PatchTST for Multiple Stock Market Prediction  

This repository  implements the **PatchTST** model for predicting multiple stock markets and cryptocurrencies. PatchTST, as introduced in [PatchTST: Time-Series Transformers for Irregular Spatiotemporal Signals](https://arxiv.org/abs/2211.14730), leverages patch-based tokenization for effective time-series forecasting.  

### Key Features:
1. **PatchTST Model:**  
   - A transformer-based model designed for time-series data.  
   - Handles irregular time-series patterns effectively.  

2. **Normalization with RevIN:**  
   - Incorporates a normalization technique inspired by [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p).  
   - RevIN significantly enhances performance by mitigating distribution shifts in time-series data.  

3. **Training on Multiple Datasets:**  
   - During training, the model is trained on multiple datasets of stock data, allowing it to generalize effectively across various markets and time-series characteristics.

### Repository Structure:
The code in this repository is adapted from the official GitHub implementations of the original papers:  

- [PatchTST](https://github.com/yuqinie98/PatchTST)  
- [RevIN](https://github.com/ts-kim/RevIN)


# To-Do:
- Masking techniques to counter the $y_t = y_{t-1}$ issue.