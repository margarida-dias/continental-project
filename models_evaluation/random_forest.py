import math

if __name__ == "__main__":

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd
    import numpy as np

    np.random.seed(42)  # For reproducibility

    n_samples = 10000

    # Generating synthetic data
    data = {
        'Tire Type': np.random.choice(['P', 'LT', 'ST', 'T'], n_samples),
        'Tire Width': np.random.randint(145, 355, n_samples),
        'Aspect Ratio': np.random.randint(40, 90, n_samples),
        'Construction': np.random.choice(['R', 'D', 'B'], n_samples),
        'Rim Diameter': np.random.randint(14, 22, n_samples),
        'Load Index': np.random.randint(75, 105, n_samples),
        'Speed Rating': np.random.choice(['S', 'T', 'U', 'H', 'V'], n_samples),
        'Season': np.random.choice(['Summer', 'Winter', 'All-Season'], n_samples),
        'Material Hardness': np.random.uniform(50, 80, n_samples),
        'Tensile Strength': np.random.uniform(10, 20, n_samples),
        'Performance Metric': np.random.uniform(70, 100, n_samples)  # Target variable
    }

    df = pd.DataFrame(data)

    df_encoded = pd.get_dummies(df, columns=['Tire Type', 'Construction', 'Speed Rating', 'Season'])

    # Splitting the dataset into features (X) and target variable (y)
    X = df_encoded.drop('Performance Metric', axis=1)
    y = df_encoded['Performance Metric']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    random_forest_model.fit(X_train, y_train)

    # Make predictions
    predictions = random_forest_model.predict(X_test)
    print(f" The performance for the tire are: {predictions}")

    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)

    # Calculate R-squared
    r2 = r2_score(y_test, predictions)

    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

"""
 The performance for the tire are: [87.04632379 86.66382271 86.21815313 ... 86.8797496  84.96497727
 84.18805944]
RMSE: 8.727032961361049
R-squared (R²): -0.02372488644618942
"""