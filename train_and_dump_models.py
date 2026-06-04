import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'Linear Regression', 'Algerian_forest_fires_cleaned_dataset.csv')

OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)


def save_obj(obj, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}. "
            f"Create it by running Linear Regression/Model_Training.ipynb (or ensure file exists)."
        )

    df = pd.read_csv(DATA_PATH)

    # Notebook preprocessing (best-effort)
    drop_cols = [c for c in ['day', 'month', 'year'] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)

    if 'Classes' in df.columns:
        df['Classes'] = np.where(df['Classes'].astype(str).str.contains('not fire'), 0, 1)

    target = 'FWI'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Feature selection by correlation (match notebook)
    corr = X_train.corr(numeric_only=True)
    threshold = 0.75
    col_corr = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                col_corr.add(cols[i])

    if col_corr:
        X_train = X_train.drop(list(col_corr), axis=1)
        X_test = X_test.drop(list(col_corr), axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'linreg': LinearRegression(),
        'lasso': Lasso(),
        'ridge': Ridge(),
        'elasticnet': ElasticNet(),
        'lassocv': LassoCV(),
    }

    metrics = {}
    fitted = {}

    for key, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        metrics[key] = {
            'mae': float(mean_absolute_error(y_test, preds)),
            'r2': float(r2_score(y_test, preds)),
        }
        fitted[key] = model

    # Save artifacts
    # Also store feature metadata so inference can rebuild consistent columns.
    feature_cols = list(X_train.columns)

    scaler_path = save_obj(scaler, 'scaler.pkl')
    save_obj(feature_cols, 'feature_columns.pkl')

    for key, model in fitted.items():
        save_obj(model, f'model_{key}.pkl')

    with open(os.path.join(OUT_DIR, 'training_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    print('Saved artifacts to:', OUT_DIR)
    print('Scaler:', scaler_path)
    for k, v in metrics.items():
        print(k, v)


if __name__ == '__main__':
    main()

