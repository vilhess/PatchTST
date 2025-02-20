# Implementation of PatchTST and FEDformer for multiple sinusoidal waves at different frequencies

This repository  implements the **PatchTST** and  **FEDformer** models for predicting a multivariate signal consisting of multiple sinusoidal waves at different frequencies.


## PatchTST:
Paper: [PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
   - A transformer-based model designed for time-series data.  
   - Leverages patch-based tokenization for effective time-series forecasting. 
   - Channel Independance
   - Incorporates a normalization technique from [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p).  
   - RevIN significantly enhances performance by mitigating distribution shifts in time-series data.  

## FEDformer:

Paper: [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://arxiv.org/abs/2201.12740) (We focus on the Fourier version)
   - A transformer-based encoder-decoder model designed for time-series data.  
   - Multi-Head Attention computed on the Fourier decomposition basis.
   - Can incorporate time-dependant features known beforehand

### Repository Structure:
The code in this repository is adapted from the official GitHub implementations of the original papers:  

- [PatchTST](https://github.com/yuqinie98/PatchTST)  
- [RevIN](https://github.com/ts-kim/RevIN)
- [FEDformer](https://github.com/MAZiqing/FEDformer)
