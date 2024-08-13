# Implementation and Fine-Tuning of a Transformer-based Model for Stock Market Prediction

### Introduction
This report details the implementation and fine-tuning of a transformer-based model designed to generate trade recommendations for the stock market. The project aimed to leverage advanced machine learning techniques to process trade and market data, ultimately providing actionable insights for trading decisions.

### Project Overview 
The task involved developing a transformer model using PyTorch, fine-tuning it on a provided dataset to forecast market trends and technical indicators, and generating Buy, Sell, or Hold signals. These predictions were then used to execute trades in a trading blotter. The objective was to leverage the transformer model to predict accurate signals, create a trading blotter, and evaluate its performance against the given simple blotter.


This repo contains the code for the BH Work Trial Task
For more information on the market data, select the TBBO schema on this page: https://databento.com/datasets/XNAS.ITCH

If you are having issues with TA Lib, Try this:
```
pip install ta-lib==0.4.0

pip install stable-baselines3==1.3.0
```
If the above commands don't work, try using `conda` for installation:
```
conda install -c conda-forge ta-lib
```

If its still not working, you can ask GPT to code each of the technical indicators. 

### Project Structure

The project repo is organized into several folders and files.

**Relevant (Important) Folders:**
- Data 
    - data.csv: Original data
    - df_with_predictions.csv: Original data with model predictions
    - trades_blotter.csv: Trades 
- Solution_notebooks
    - TEncoder_solution.ipynb: Contains the implementation of the transformer encoder with a classification head architecture. 
    - trading_blotter.ipynb: Contains the implementation of trading environment
- Models
    - best_model_checkpoint.pth: Weights of the model stored after early stopping
    - transformer_agent.pth: Weights of the model after full training (20 epochs)

**Not so Important Folders:**
- Research
    - research.ipynb: Contains feature engineering, implementation of PCA and t-SNE (along with visualizations) for dimensionality reduction and to check if the data had any clusters for label generation. 
    - TST.py: Contains the implementation of Time Series Transformer (used mainly for time series forecasting). This architecture was abandoned due to its poor performance even after a lot of fine-tuning. 
- Simple_blotters
    - Given implementation of trading blotter with PPO
    - simple_blotter.ipynb: Updated version of the given notebook with calculated metrics
- Files: 
    - requirements.txt: All the used packages in the project
    - BH Report.pdf: Documentation of the project



