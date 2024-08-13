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
If its still not working, you can ask GPT to code each of the technical indicators. 

You do not have to create a transformer from scratch, feel free to use a pretrained model and fine tune it


