# Electricity-Load
Predict Electricity load from historical time series.The data set is available from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. This computes random cauchy projections as part of feature engineering and uses the past 24 hour load as lagged variables to make predictions.

The model us built using boosted trees(xgboost) and is competetive with state of the art time series forecasting methods built on the same dataset.

1. Matrix factorization applied to time series : https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
2. Time series forecasting using Deep RNNs: https://arxiv.org/pdf/1704.04110.pdf


