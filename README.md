# Energy Production Prediction using Time Series Data

## Project Overview
This project focuses on forecasting energy production (specifically from solar and wind sources) using machine learning techniques on time series data. The dataset contains temporal features and binary indicators for different energy sources. The goal is to develop models that can accurately predict hourly energy production.

## Key Features
- **Data Preprocessing**: Includes handling missing values, feature engineering (such as extracting day, month, season, etc.), and scaling.
- **Exploratory Data Analysis (EDA)**: Detailed analysis of energy production trends, seasonal patterns, and correlations between features.
- **Modeling**: Baseline model (Random Forest Regressor) and deep learning model (LSTM with dense layers) used for prediction.
- **Hyperparameter Tuning**: Keras Tuner was used to optimize the deep learning model for better performance.
- **Prediction & Results**: Accurate prediction of energy production values for solar and wind energy sources with model validation.

## Table of Contents
1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Models and Training](#models-and-training)
5. [Future Work](#future-work)
6. [Contributing](#contributing)

## Dataset
The dataset includes time-based attributes and energy production values, broken down by hour and source.

### Columns:
- `Date` & `Hour`: Time information.
- `Production`: Energy production values.
- `dayOfYear`, `dayName`, `monthName`, etc.: Extracted time features.
- `Source_Wind`: Binary column indicating wind and solar energy sources.
- `Production_scaled`: Scaled production data for modeling.
- `Source_encoded`, `Season_encoded`: Encoded categorical features for model input.

## Installation

### Requirements
- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Keras Tuner

Install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/energy-production-prediction.git
   ```

2. Navigate to the project directory and preprocess the data:

   ```bash
   python preprocess.py
   ```

3. Train the model:

   ```bash
   python train_model.py
   ```

## Models and Training

### Baseline Model
- **Random Forest Regressor** used as a baseline for performance comparison.

### Deep Learning Model (LSTM)
- **LSTM** (Long Short-Term Memory) architecture trained with time-based features.
- **Dropout layers** used for regularization and **ReLU activation** in dense layers.

### Hyperparameter Tuning
- **Keras Tuner** was used to optimize the LSTM model by tuning units in LSTM layers, dense layers, dropout rates, and learning rate.

## Future Work
Potential future improvements include:
- Incorporating **weather data** to further enhance prediction accuracy.
- Experimenting with advanced architectures like **Bidirectional LSTMs** or **Transformer-based models**.
- Exploring other hyperparameter tuning methods like **Bayesian Optimization**.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or raise an issue to discuss potential changes or improvements.
