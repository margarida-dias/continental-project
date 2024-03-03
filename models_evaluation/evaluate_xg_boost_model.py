import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    import xgboost as xgb

    np.random.seed(42)  # For reproducibility

    n_samples = 10000

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost regressor with default parameters
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

    kfold = KFold(shuffle=True, random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
    }

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

    # Predictions on training and test sets
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    # Calculate metrics
    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    test_rmse = mean_squared_error(y_test, test_preds, squared=False)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    print(f"Training R²: {train_r2}, Test R²: {test_r2}")

"""
Fitting 5 folds for each of 27 candidates, totalling 135 fits
Best Parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
Best Score: 8.586905607045495
Training RMSE: 8.55784092601975, Test RMSE: 8.62020743872186
Training R²: 0.0048389680880758235, Test R²: 0.0011840653238875953

The model might be slightly underfitting, given the low R² scores, suggesting 
it's not capturing the complexity of the data well enough to explain a larger portion 
of the variance in the target variable.
"""