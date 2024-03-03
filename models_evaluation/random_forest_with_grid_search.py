
if __name__ == "__main__":

    import math
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Define the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Set up the grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Use the best model to make predictions
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    # Calculate Mean Squared Error, then take the square root to get RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)

    # Calculate R-squared
    r2 = r2_score(y_test, predictions)

    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

"""
Best parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}
RMSE: 8.633941232713637
R-squared (R²): -0.0020011155037451545
"""