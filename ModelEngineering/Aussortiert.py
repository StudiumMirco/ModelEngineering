from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
from statsmodels.tsa.vector_ar.var_model import VAR
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
import pmdarima as pm
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import model_selection
import skforecast



# Merkmalserzeugeung, Merkmalsuswahl  und Aufteilen der Daten in Trainings- und Testsets mit tsfresh
# X_train, X_test, y_train, y_test = tsfresh_featureextraction(df)

#def tsfresh_featureextraction(df):
#     print(df.info())
#     print(df.head())
#
#     #Lags einrichten:
#     for lag in range(360,370):
#         df[f'sby_need_lag_{lag}'] = df['sby_need'].shift(lag)
#         df[f'n_sick_lag_{lag}'] = df['n_sick'].shift(lag)
#         df[f'calls_lag_{lag}'] = df['calls'].shift(lag)
#         df[f'n_duty_lag_{lag}'] = df['n_duty'].shift(lag)
#
#     df.dropna(inplace =True)
#
#     print(df.info())
#     print(df.head())
#      # Features und Zielvariable
#
#
#
#     # Originalen Datumsindex speichern
#     original_dates = df.index
#     # Erstellen einer ID-Spalte für tsfresh
#     df['id'] = 1 # Setzen Sie für alle Zeilen dieselbe ID, wenn es sich um eine einzelne Zeitreihe handelt
#     df.reset_index(inplace=True)  # Datum als Spalte für tsfresh
#
#     # Umbenennen der Spalten für tsfresh (optional, aber kann hilfreich sein)
#     df.rename(columns={'date': 'time'}, inplace=True)
#
#     # Formatieren der Daten für tsfresh
#     lag=360
#     df_tsfresh = df[['id', 'time']]
#     print(df_tsfresh.info())
#     print(df_tsfresh.head())
#     for lag in range(360,370):
#         df_tsfresh= pd.concat([df_tsfresh,df[[f'sby_need_lag_{lag}',f'n_sick_lag_{lag}',f'calls_lag_{lag}',f'n_duty_lag_{lag}']]],axis = 1)
#
#     print(df_tsfresh.info())
#     print(df_tsfresh.head())
#     # Feature-Extraktion
#     extracted_features = extract_features(df_tsfresh, column_id='id', column_sort='time')
#
#     # Zuerst die fehlenden Werte in den extrahierten Features auffüllen
#     impute(extracted_features)
#
#     # Zielvariable definieren
#     y = df['sby_need']
#
#     if len(extracted_features) != len(y):
#         raise ValueError(f"Anzahl der Zeilen stimmt nicht überein: {len(extracted_features)} != {len(y)}")
#
#     # Feature-Auswahl
#     selected_features = select_features(extracted_features, y)
#
#     #print(selected_features)
#
#
#     X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, shuffle=False)
#     # Datum für die Testdaten wiederherstellen
#     original_dates_train = original_dates[X_train.index]
#     original_dates_test = original_dates[X_test.index]
#
#     # # Wiederherstellen des Datumsindex für die Testdaten
#     X_train.index = original_dates_train
#     X_test.index = original_dates_test
#     y_train.index = original_dates_train
#     y_test.index = original_dates_test
#
#     return X_train, X_test, y_train, y_test
#
#     #eturn selected_features

# def var_prediction(X_train, X_test, y_train, y_test, params):
#
#     # Kombinieren von X_train und y_train zu einem DataFrame
#     train_data = pd.concat([X_train, y_train], axis=1)
#
#     # Erstellen und Trainieren des VAR-Modells
#     model_var = VAR(endog=train_data)
#     model_var_fit = model_var.fit(maxlags=params)  # Du kannst den Wert von maxlags je nach Bedarf anpassen
#
#     # Vorhersage für den Testzeitraum
#     lag_order = model_var_fit.k_ar
#     input_data = train_data.values[-lag_order:]  # Letzte Beobachtungen verwenden, um die Vorhersage zu starten
#     var_predictions = model_var_fit.forecast(input_data, steps=len(y_test))
#
#     # Konvertieren der Vorhersagen in einen DataFrame mit den gleichen Spaltennamen wie in train_data
#     var_predictions_df = pd.DataFrame(var_predictions, index=X_test.index, columns=train_data.columns)
#
#     # Extrahieren der Vorhersagen für y
#     y_pred_var = var_predictions_df[y_train.name]
#
#     return y_pred_var



# !Nicht mehr verwendet!

# def arima_prediction(train):
#
#     # Train-Test-Split für Modellvalidierung (optional)
#     #train, test = model_selection.train_test_split(df['sby_need'], train_size=0.8)
#
#
#     # # Automatische ARIMA-Modellauswahl
#     # model = pm.auto_arima(
#     #     train['sby_need'],
#     #     start_p=1, start_q=1,
#     #     max_p=5, max_q=5,
#     #     start_P=0, start_Q=0,
#     #     max_P=2, max_Q=2,
#     #     m=30,  # Jährliche Saisonalität
#     #     seasonal=True,
#     #     d=None,  # Automatische Differenzierung
#     #     D=1,    # Einmalige saisonale Differenzierung
#     #     trace=True,
#     #     error_action='ignore',
#     #     suppress_warnings=True,
#     #     stepwise=True
#     # )
#
#
#     # # Festlegen der SARIMA-Parameter (z.B. p=2, d=1, q=2; P=1, D=1, Q=1)
#     model = SARIMAX(train['sby_need'],
#                     exog=train[['weekday', 'week_of_year', 'month']],
#                     order=(1, 0, 1),
#                     seasonal_order=(1, 0, 1, 365))
#
#     model_fit = model.fit(disp=False)
#         # Vorhersage für die nächsten 30 Tage
#     forecast = model_fit.forecast(steps=30, exog=train[['weekday', 'week_of_year', 'month']])
#
#     # Vorhersagen anzeigen
#     print(forecast)
#
#
#
#     # # Zusammenfassung des besten Modells
#     # print(model.summary())
#     #
#     # # Vorhersage der nächsten 30 Tage
#     # forecast = model.predict(n_periods=30)
#     #
#     # # Vorhersagen anzeigen
#     # print(forecast)
#     return forecast
# #
# def test(df):
#     # Daten laden
#
#     # Zielvariable
#     y = df['calls']
#
#     # Prädiktoren
#    #X = df[[f'sby_need_lag_365', 'weekday', 'month', 'week_of_year']]
#
#     # Automatische Lag-Auswahl mit pmdarima
#     # Hier wird auto_arima verwendet, um auch externe Regressoren (X) und Lags zu optimieren
#     model = pm.auto_arima(y,
#                           #exogenous=X,
#                           start_p=1, start_q=1,
#                           max_p=5, max_q=5,
#                           m=365,  # Angabe der saisonalen Periode
#                           seasonal=True,
#                           d=None,
#                           trace=True,
#                           error_action='ignore',
#                           suppress_warnings=True,
#                           stepwise=True,
#                           n_fits=50)  # Begrenzung der Modellanpassungen, um Laufzeit zu reduzieren
#
#     # Zusammenfassung des besten Modells
#     print(model.summary())
#
#     # Vorhersage der nächsten 30 Tage
#     forecast = model.predict(n_periods=30)  #exogenous=X[-30:]
#
#     # Vorhersagen anzeigen
#     print(forecast)
#     return forecast

# def sarima_prediction(y_train, y_test):
#     # Verwenden der 'sby_need'-Spalte für die Modellierung
#     y = y_train
#
#     # Definition des SARIMA-Modells
#     # Annahme: p, d, q = 1, 1, 1 und P, D, Q = 1, 1, 1 für jährliche Saisonalität
#     # m = 365 für tägliche Daten mit jährlicher Saisonalität
#     sarima_model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
#
#     # Fitten des Modells
#     sarima_fit = sarima_model.fit(disp=False)
#
#     # Vorhersage auf dem Testset (z.B. die letzten 365 Tage)
#     # Anzahl der Tage für die Vorhersage
#     forecast = sarima_fit.get_forecast(steps=len(y_test))
#     forecast_values = forecast.predicted_mean
#
#     return forecast_values
