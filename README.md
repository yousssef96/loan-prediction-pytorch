# Loan Approval Prediction with PyTorch Lightning

This project implements a machine learning pipeline to predict loan approval using a Multi-Layer Perceptron (MLP) built with PyTorch Lightning. The pipeline handles preprocessing, model training, evaluation, and visualization of results.

## Key Features

### Data Preprocessing
- Normalization of numerical features using `StandardScaler`.
- Encoding of categorical variables using `LabelEncoder` and one-hot encoding.
- Train-test split with validation set for model evaluation.

### Neural Network Model
- Multi-layer perceptron with configurable hidden units.
- Binary classification with `binary_cross_entropy_with_logits` loss.
- Metrics tracked: accuracy, confusion matrix, and optional additional metrics.

### Training and Evaluation
- Early stopping and model checkpointing with `ModelCheckpoint`.
- Evaluation on test data with metrics logging.
- Visualization of training and validation loss and accuracy.

### Exploratory Data Analysis
- Distribution plots for numerical features.
- Mosaic plots and chi-square tests for categorical features.

## Results
The model achieved a **test accuracy of 91.64%** and a **test loss of 0.1872**, demonstrating effective prediction of loan approval status. Confusion matrices and other metrics provide detailed performance insights.
