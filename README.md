This repo contains the code for the BH Work Trial Task
For more information on the market data, select the TBBO schema on this page: https://databento.com/datasets/XNAS.ITCH

If you are having issues with TA Lib, Try this:

pip install ta-lib==0.4.0

pip install stable-baselines3==1.3.0

If its still not working, you can ask GPT to code each of the technical indicators. 

You do not have to create a transformer from scratch, feel free to use a pretrained model and fine tune it

Transformer Ecoder With Classification Head:

1) Added Classification Head to a traditoinal transformer encoder.
2) Added learning rate scheduler (StepLR)
3) Implemented Early Stopping
4) Added Dropout layers and L2 regularization
5) Handled imbalance classes in the data

Test accuracy without any above modifications: around 91%
Test accuracy with all the above modifications: around 81%
Test accuracy with class imbalance: 82
Test accuracy without dropout layers: 83
Test accuracy without dropout and L2: 82
Test accuracy without dropout and L2 and lr scheduler: 
