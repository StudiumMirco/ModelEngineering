from skforecast.model_selection import grid_search_forecaster
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.ForecasterRnn import ForecasterRnn
from skforecast.ForecasterRnn.utils import create_and_compile_model
from sklearn.preprocessing import MinMaxScaler
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
import matplotlib.pyplot as plt
from skforecast.model_selection import backtesting_forecaster
from sklearn.linear_model import Ridge
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.preprocessing import StandardScaler
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA


# Vorhersage Baselinemodel
def baseline_seasonal_naive_prediction(y_train, y_test):
    # Baselinemodell
    seasonal_lag = 365  # Bei täglicher Datenfrequenz

    # Saisonales Naives Modell (angenommen jährliche Saisonalität)
    # Nur die letzten Werte auswählen, die der Länge von y_test entsprechen
    seasonal_naive_forecast = y_train.iloc[-seasonal_lag:-(seasonal_lag - len(y_test))]

    # Index des Testdatensatzes übernehmen, um sicherzustellen, dass die Datumsbereiche übereinstimmen
    seasonal_naive_forecast.index = y_test.index

    return seasonal_naive_forecast


# Erstellen LGBM-Forecaster
def lightgbm():
    # Forecaster erstellen
    forecaster = ForecasterAutoreg(
        regressor=LGBMRegressor(random_state=15926, verbose=-1),
        lags=365
    )
    return forecaster


# Vorhersage mit LGBM mit Hyperparametertuning und Backtesting
def lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test):
    # Hyperparametersuche Lightgbm mit Gridsearch
    # ==============================================================================

    forecasterlgbm = lightgbm()
    # Lags
    lags_grid = {
        'lags_1': 3,
        'lags_2': 10,
        'lags_3': 30,
        'lags_4': 365,
        'lags_5': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        'lags_6': 60,
        'lags_7': [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 363, 364, 365, 366, 367],
        'lags_8': [1, 2, 3, 4, 5, 30, 60, 90, 365]
    }

    # Regressor Hyperparameter
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [5, 10, 15, 20]
    }

    # Durchführen der Gridsearch zu Hyperparameteroptimierung
    results = grid_search_forecaster(
        forecaster=forecasterlgbm,
        y=y_trainval,
        exog=X_trainval,
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=36,
        refit=True,
        metric='mean_absolute_error',
        initial_train_size=y_train.size,
        fixed_train_size=False,
        return_best=True,
        n_jobs='auto',
        verbose=False,
        show_progress=True
    )

    # Vorhersage mit Backtesting
    metric, prediction = backtesting(pd.concat([y_trainval, y_test]), forecasterlgbm, y_trainval.size,
                                     pd.concat([X_trainval, X_test]))
    return metric, prediction, forecasterlgbm


# Bestimmen eines Konfidenzintervalls von LGBM-Modell
def probalistic_lgbm(forecasterlgbm, predictionlgbm, target, X, y_test, train_size, up):
    # Perzentilbestimmung mit LGBM
    # Residuen bestimmen
    residuals = y_test - predictionlgbm['pred']
    residuals = residuals.dropna()

    forecasterlgbm.set_out_sample_residuals(residuals=residuals, y_pred=y_test)

    # Backtesting mit Vorhersageintervallen unter Benutzung der Out-Sample-Residuals

    metric, intervals = backtesting_forecaster(
        forecaster=forecasterlgbm,
        y=target,
        exog=X,
        steps=35,
        metric='mean_absolute_error',
        initial_train_size=train_size,
        refit=True,
        fixed_train_size=False,
        interval=[10, up],
        n_boot=200,
        in_sample_residuals=False,  # out-sample residuals
        binned_residuals=True,
        n_jobs='auto',
        verbose=False,
        show_progress=True
    )

    return metric, intervals


# Erstellen eines XGBoost Forecasters
def xgboost():
    # Erstellen XGBoost-Forecaster
    forecaster = ForecasterAutoreg(
        regressor=XGBRegressor(tree_method='hist', random_state=123),
        lags=365,
    )
    return forecaster


# Vorhersage mit XGBoost mit Hyperparametertuning und Backtesting
def xgboost_prediction(X_trainval, X_test, y_trainval, y_train, y_test):
    forecasterxgb = xgboost()

    # Hyperparametersuche XGBoost mit Gridsearch
    # Lags
    lags_grid = {
        'lags_1': 3,
        'lags_2': 10,
        'lags_3': 30,
        'lags_4': 365,
        'lags_5': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        'lags_6': 60,
        'lags_7': [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 363, 364, 365, 366, 367],
        'lags_8': [1, 2, 3, 4, 5, 30, 60, 90, 365]
    }

    # Regressor Hyperparameter
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [5, 10, 15, 20]
    }
    # Hyperparametertuning
    results = grid_search_forecaster(
        forecaster=forecasterxgb,
        y=y_trainval,
        exog=X_trainval,
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=36,
        refit=True,
        metric='mean_absolute_error',
        initial_train_size=y_train.size,
        fixed_train_size=False,
        return_best=True,
        n_jobs='auto',
        verbose=False,
        show_progress=True
    )

    print(results)

    # Vorhersage mit Backtesting
    metric, prediction = backtesting(pd.concat([y_trainval, y_test]), forecasterxgb, y_trainval.size,
                                     pd.concat([X_trainval, X_test]))
    return metric, prediction


# Erstellen eines RandomForest-Forecasters
def rfregressor():
    # Erstellen Forecaster
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=365,
    )
    return forecaster


# Vorhersage mit RandomForest mit Hyperparametertuning und Backtesting
def rfregressor_prediciton(X_trainval, X_test, y_trainval, y_train, y_test):
    forecasterrf = rfregressor()

    # Hyperparametersuche RFRegressor mit Gridsearch
    # Lags
    lags_grid = {
        'lags_1': 3,
        'lags_2': 10,
        'lags_3': 30,
        'lags_4': 365,
        'lags_5': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        'lags_6': 60,
        'lags_7': [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 363, 364, 365, 366, 367],
        'lags_8': [1, 2, 3, 4, 5, 30, 60, 90, 365]
    }

    # Regressor Hyperparameter
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [5, 10, 15, 20]
    }
    # Hyperparametertuning
    results = grid_search_forecaster(
        forecaster=forecasterrf,
        y=y_trainval,
        exog=X_trainval,
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=36,
        refit=True,
        metric='mean_absolute_error',
        initial_train_size=y_train.size,
        fixed_train_size=False,
        return_best=True,
        n_jobs='auto',
        verbose=False,
        show_progress=True
    )

    print(results)
    # Vorhersage mit Backtesting
    metric, prediction = backtesting(pd.concat([y_trainval, y_test]), forecasterrf, y_trainval.size,
                                     pd.concat([X_trainval, X_test]))
    return metric, prediction


# Direct-Multistep Regression mit Ridge
def ridge_prediction(data_train):
    # Forecaster erzeugen
    forecaster = ForecasterAutoregDirect(
        regressor=Ridge(random_state=123),
        steps=36,
        lags=8,
        transformer_y=StandardScaler()
    )
    # Hypertuning
    param_grid = {'alpha': np.logspace(-5, 5, 10)}
    lags_grid = {
        'lags_1': 3,
        'lags_2': 10,
        'lags_3': 30,
        'lags_4': 365,
        'lags_5': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
        'lags_6': 60,
        'lags_7': [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 363, 364, 365, 366, 367],
        'lags_8': [1, 2, 3, 4, 5, 30, 60, 90, 365]
    }
    results_grid = grid_search_forecaster(
        forecaster=forecaster,
        y=data_train['sby_need'],
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=36,
        refit=False,
        metric='mean_squared_error',
        initial_train_size=int(len(data_train) * 0.8),
        fixed_train_size=False,
        return_best=True,
        n_jobs='auto',
        verbose=False
    )
    # Vorhersage
    predictions = forecaster.predict()
    return predictions


# Backtesting für die rekursiven Multistep Modelle
def backtesting(target, forecaster, train_size, X):
    # Backtest Model on test data
    metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=target,
        exog=X,
        steps=36,
        metric='mean_absolute_error',
        initial_train_size=train_size,
        refit=True,
        fixed_train_size=False,
        skip_folds=None,
        n_jobs='auto',
        verbose=True,
        show_progress=True
    )
    return metric, predictions


# Direct- Multistep-Vorhersage mit Random Forest
def random_forest_prediction(X_trainval, X_train, X_val, X_test, y_trainval, y_train, y_val):
    # Erstellen der Gitter für die Gittersuche nach den besten Parametern
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    # Grid-Search for Random Forest
    best_mae = float('inf')
    best_params_RF = None

    # Hyperparametertuning
    for params in ParameterGrid(rf_param_grid):
        # Trainieren eines  Random Forest-Modells
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Vorhersagen
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)

        if mae < best_mae:
            best_mae = mae
            best_params_RF = params

    print(f"Best Random Forest Parameters: {best_params_RF}, MAE: {best_mae}")

    # Trainieren eines  Random Forest-Modells
    model = RandomForestRegressor(**best_params_RF)
    model.fit(X_trainval, y_trainval)

    # Vorhersagen
    predictions = model.predict(X_test)

    return predictions


# Direct- Multistep-Vorhersage mit Support Vector Regressor
def svr_prediction(X_trainval, X_train, X_val, X_test, y_trainval, y_train, y_val):
    # Erstellen der Gitter für die Gittersuche nach den besten Parametern
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear']
    }

    # Grid-Search für SVR
    best_mae = float('inf')
    best_params_svr = None

    # Hyperparametertuning
    for params in ParameterGrid(svr_param_grid):
        # Modellinitialisierung
        model_svr = SVR(**params)
        # Modelltraining
        model_svr.fit(X_train, y_train)
        # Vorhersagen
        svr_predictions = model_svr.predict(X_val)

        mae = mean_absolute_error(y_val, svr_predictions)

        if mae < best_mae:
            best_mae = mae
            best_params_svr = params

    print(f"Best SVR Parameters: {best_params_svr}, MAE: {best_mae}")

    # SVR
    # Modellinitialisierung
    model_svr = SVR(**best_params_svr)
    # Modelltraining
    model_svr.fit(X_trainval, y_trainval)
    # Vorhersagen
    svr_predictions = model_svr.predict(X_test)

    return svr_predictions


## Vorhersage mit neuronalem Netzwerk(NN) mit Hyperparametertuning und Backtesting
def lstm_prediction(y_train, y_val, y_trainval, data, lags, y_name):
    # Anpassen der Daten auf die Vorraussetzungen von LSTN
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    data = data.astype('float32')

    # Erstellen des LSTM-Netzwerks
    model = create_and_compile_model(
        series=pd.DataFrame(y_train),
        levels=y_name,
        lags=lags,
        steps=36,
        recurrent_layer="LSTM",
        recurrent_units=[100, 50],
        dense_units=[64, 32],
        optimizer=Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )
    # Erzeugen des Forecasters
    forecaster = ForecasterRnn(
        regressor=model,
        levels=y_name,
        steps=36,
        lags=lags,
        transformer_series=MinMaxScaler(),
        fit_kwargs={
            "epochs": 10,  # Number of epochs to train the model.
            "batch_size": 32,  # Batch size to train the model.
            "callbacks": [
                EarlyStopping(monitor="val_loss", patience=5)
            ],  # Callback to stop training when it is no longer learning.
            "series_val": y_val,  # Validation data for model training.
        },
    )
    # Fit
    forecaster.fit(y_train)

    # fig, ax = plt.subplots(figsize=(5, 2.5))
    # forecaster.plot_history(ax=ax)

    # Vorhersage und Backtesting
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=forecaster,
        steps=forecaster.max_step,
        series=pd.DataFrame(data),
        levels=forecaster.levels,
        initial_train_size=y_trainval.size,
        metric="mean_absolute_error",
        verbose=False,
        refit=False,
    )

    return metrics, predictions


# Arima-Modell - Nicht mehr verwendet, da nicht geeignet (Speicher läuft über)
def arima_model(train):
    # Train-Test-Split für Modellvalidierung (optional)
    # train, test = model_selection.train_test_split(df['sby_need'], train_size=0.8)

    # Automatische ARIMA-Modellauswahl
    model = pm.auto_arima(
        train,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        m=365,  # Jährliche Saisonalität
        seasonal=True,
        d=None,  # Automatische Differenzierung
        D=1,  # Einmalige saisonale Differenzierung
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print(model)


# Arima-Vorhersage - Nicht mehr verwendet, da nicht geeignet (Speicher läuft über)
def arima_prediction(train, test):
    sarima = ARIMA(train, order=(1, 0, 1), seasonal_order=(0, 1, 0, 365)).fit()
    print(sarima.summary())

    prediction = pd.DataFrame(sarima.predict(n_periods=len(test)), index=test.index)
    print(prediction)
    prediction.columns = ['pred']
    print(prediction)

    return prediction
