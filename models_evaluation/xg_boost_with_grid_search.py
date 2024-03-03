from deployment.core.config import settings

if __name__ == "__main__":

    import xgboost as xgb
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    import pandas as pd
    import numpy as np

    np.random.seed(42)  # For reproducibility

    n_samples = 100000

    # Generating synthetic data
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

    df_encoded = pd.get_dummies(df, columns=['Tire Type', 'Construction', 'Speed Rating', 'Season'])

    # Splitting the dataset into features (X) and target (y)
    X = df_encoded.drop('Performance Metric', axis=1)
    y = df_encoded['Performance Metric']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

    # Define the parameter grid to search
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    kfold = KFold(shuffle=True, random_state=42)

    # Set up the grid search
    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               cv=kfold,
                               scoring='neg_mean_squared_error',
                               verbose=1)

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", np.sqrt(-grid_search.best_score_))

    # Use the best model
    best_model = grid_search.best_estimator_

    # Predictions
    predictions = best_model.predict(X_test)
    print(f" The performance for the tire are: {predictions}")

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Test RMSE: {rmse}")
    print(f"Test R-squared: {r2}")

    best_model.save_model(settings.MODEL_URI)


"""
Fitting 3 folds for each of 27 candidates, totalling 81 fits
Best Parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
Best Score: 8.584268823260654
The performance for the tire are: [85.07224  84.971596 85.06037  ... 85.290764 85.07224  85.082855]
RMSE: 8.62020743872186
R-squared: 0.0011840653238875953
"""
