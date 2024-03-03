from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import Ridge
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

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline for preprocessing and modeling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    # Parameters for Grid Search
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Grid Search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Making predictions & evaluating the model
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

"""
Fitting 5 folds for each of 6 candidates, totalling 30 fits
Best parameters: {'model__alpha': 100}
Mean Squared Error (MSE): 75.18205572228558
Root Mean Square Error (RMSE): 8.670758658980516
R-squared (R²): -0.003702920610257454
"""