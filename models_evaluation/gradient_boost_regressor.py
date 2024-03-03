
if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    np.random.seed(42)  # For reproducibility

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

    # Defining features (X) and target (y)
    X = df_encoded.drop('Performance Metric', axis=1)
    y = df_encoded['Performance Metric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Gradient Boosting Regressor
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    print(f" The performance for the tire are: {predictions}")

    # Calculate the Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Square Error: {rmse}")

    from sklearn.metrics import r2_score

    # Calculate R-squared
    r2 = r2_score(y_test, predictions)
    print(f"R-squared: {r2:.2f}")

"""
The performance for the tire are: [85.20677462 84.94403325 84.58349396 ... 86.23923125 85.29727049
84.98847679]
RMSE: 8.627218510427094
R-squared: -0.00
"""