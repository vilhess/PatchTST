# Implementation of PatchTST for multiple sinusoidal waves at different frequencies

This repository  implements the **PatchTST** model for predicting a multivariate signal consisting of multiple sinusoidal waves at different frequencies. PatchTST, as introduced in [PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730), leverages patch-based tokenization for effective time-series forecasting.  

### Key Features:
1. **PatchTST Model:**  
   - A transformer-based model designed for time-series data.  
   - Handles irregular time-series patterns effectively.  

2. **Normalization with RevIN:**  
   - Incorporates a normalization technique inspired by [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p).  
   - RevIN significantly enhances performance by mitigating distribution shifts in time-series data.  

### Repository Structure:
The code in this repository is adapted from the official GitHub implementations of the original papers:  

- [PatchTST](https://github.com/yuqinie98/PatchTST)  
- [RevIN](https://github.com/ts-kim/RevIN)
