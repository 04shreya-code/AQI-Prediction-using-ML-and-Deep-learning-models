# AQI-Prediction-using-ML-and-Deep-learning-models

# Overview
This project aims to predict Air Quality Index (AQI) using machine learning models trained on environmental data. The AQI prediction is crucial for understanding air pollution levels, which impact public health and environmental policies.

# Features
The prediction models use the following features extracted from environmental datasets:

## Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene
Environmental Factors: Region, Day period, Month encoded, Season, Weekday or weekend, Regular day or holiday

# Models Implemented
## CNN-LSTM
Combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for capturing temporal dependencies.
#### Strength: Effective in capturing complex spatial and temporal patterns in pollutant data.
#### Implementation: Implemented using TensorFlow/Keras.

## XGBoost-CNN-LSTM
Integrates Gradient Boosting (XGBoost) to enhance feature extraction capabilities before feeding into a CNN-LSTM model.
#### Strength: Improves feature handling and prediction accuracy through boosted trees and neural networks.
#### Implementation: XGBoost for feature extraction followed by TensorFlow/Keras for CNN-LSTM.

## BiGRU-AM
Utilizes Bidirectional Gated Recurrent Units (BiGRU) with an attention mechanism (AM) to focus on important input features.
#### Strength: Enhances context understanding and feature relevance, improving prediction performance.
#### Implementation: Implemented using TensorFlow/Keras with custom attention layers.

## Random Forest
A robust ensemble learning method that constructs multiple decision trees to predict the AQI.
#### Strength: Handles non-linear relationships and complex interactions between features effectively.
#### Implementation: Implemented using scikit-learn.

## PC-ANN
It Uses Principal Component Analysis (PCA) for dimensionality reduction followed by Artificial Neural Networks (ANN) for prediction.
#### Strength: Reduces computational complexity while preserving important patterns in the data.
#### Implementation: Implemented using scikit-learn and TensorFlow/Keras.

# Evaluation
Each model was trained and evaluated based on metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared to assess prediction accuracy.
Cross-validation and hyperparameter tuning were performed to optimize each model's performance.

# Conclusion
The best model for AQI prediction depends on specific requirements such as prediction accuracy, computational resources, and interpretability. Consider evaluating each model on your dataset to determine the most suitable approach for your application.
