if __name__ == "__main__":

    import xgboost as xgb
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

    # Splitting the dataset into features (X) and target (y)
    X = df_encoded.drop('Performance Metric', axis=1)
    y = df_encoded['Performance Metric']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the dataset into DMatrix object, which is optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost model parameters
    params = {
        'max_depth': 20,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Train the model
    num_boost_round = 200
    model = xgb.train(params, dtrain, num_boost_round, evals=[(dtest, 'test')], early_stopping_rounds=10)

    bst_model = xgb.XGBRegressor()

    # Predictions
    predictions = model.predict(dtest)
    print(f" The performance for the tire are: {predictions}")

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"R-squared: {r2}")

    #model.save_model('/Users/margarida/PycharmProjects/continental_project/model.json')

"""
The performance for the tire are: [84.987114 85.06621  84.94868  ... 85.20192  85.08243  84.873665]
RMSE: 8.624003517962567
R-squared: 0.00030417483133104994
"""