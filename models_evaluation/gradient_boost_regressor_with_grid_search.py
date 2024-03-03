
if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    n_samples = 10000

    # Creating a dataset
    data = {
        'Tire Type': np.random.choice(['P', 'LT', 'ST', 'T'], n_samples),
        'Tire Width (mm)': np.random.randint(145, 355, n_samples),
        'Aspect Ratio (%)': np.random.randint(40, 90, n_samples),
        'Construction': np.random.choice(['R', 'D', 'B'], n_samples),
        'Rim Diameter (inches)': np.random.randint(14, 22, n_samples),
        'Load Index': np.random.randint(75, 105, n_samples),
        'Speed Rating': np.random.choice(['S', 'T', 'U', 'H', 'V'], n_samples),
        'Season': np.random.choice(['Summer', 'Winter', 'All-Season'], n_samples),
        'Material Hardness': np.random.uniform(50, 80, n_samples),
        'Tensile Strength': np.random.uniform(10, 20, n_samples),
        'Performance Metric': np.random.uniform(70, 100, n_samples)  # Target variable
    }

    df = pd.DataFrame(data)

    # Applying one-hot encoding to categorical variables
    df_encoded = pd.get_dummies(df, columns=['Tire Type', 'Construction', 'Speed Rating', 'Season'])

    # Splitting the dataset into features (X) and target (y)
    X = df_encoded.drop('Performance Metric', axis=1)
    y = df_encoded['Performance Metric']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
    }

    # Initialize the base model
    gb_model = GradientBoostingRegressor(random_state=42)

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(estimator=gb_model,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=2,
                               scoring='neg_mean_squared_error')

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Best model after grid search
    best_model = grid_search.best_estimator_

    # Predictions with the best model
    predictions = best_model.predict(X_test)
    print(f" The performance for the tire are: {predictions}")

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Root Mean Square Error: {rmse}")
    print(f"R-squared: {r2}")
    print(f"Best Parameters: {grid_search.best_params_}")

"""
RMSE: 8.785140335185371
R-squared: -0.0014934211139643327
Best Parameters: {'learning_rate': 0.01, 'max_depth': 3, 'min_samples_split': 4, 'n_estimators': 100}
"""