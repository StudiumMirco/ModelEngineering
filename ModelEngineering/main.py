from methoden import data_preparation
from methoden import featureextraction
from methoden import metrics, featureextraction_lags, featureextraction_sby, residuals_histo, residuals_scatter, \
    feature_importance
from models import baseline_seasonal_naive_prediction, lightgbm_prediction, xgboost_prediction, rfregressor_prediciton, \
    random_forest_prediction, svr_prediction, lstm_prediction, probalistic_lgbm, ridge_prediction, arima_prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Warnung deaktivieren
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


def main():
    # Bestimmung sby_need mit besten Modell
    # Sby_need direkt bestimmen
    df = data_preparation('sickness_table.csv')

    # Direkte Vorhersage von sby_need zum Vergleich
    # print('LightGBM:')
    # metric_lgbm, prediction_sby, forecasterlgbm = lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test)
    # print(metric_lgbm)
    # metrics(prediction_sby['pred'], y_test)

    # print('Probabilistic:')
    # metric, intervals_sby = probalistic_lgbm(forecasterlgbm, prediction_sby, pd.concat([y_trainval, y_test]),
    #                                      pd.concat([X_trainval, X_test]), y_test, y_trainval.size)
    # metrics(intervals_sby['pred'], y_test)
    # metrics(intervals_sby['upper_bound'], y_test)

    # Bestimmen sby_need über calls und n_sick
    # Vorhersage Calls
    y_name = 'calls'

    # Eigene Merkmalserzeugung und Auswahl
    X_trainval, X_test, y_trainval, y_test = featureextraction(df, y_name)

    # Erneutes Aufteilen der Trainingsdaten on Trainingsdaten und Validierungsdaten
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

    # Vorhersage der Anzahl der Anrufe mit LGBM
    metric_lgbm, prediction_calls, forecasterlgbm = lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test)

    # Feature Importance bei der Calls Vorhersage
    feature_importance(forecasterlgbm)

    # Ergebnisse vorbereiten zur Weiterverarbeitung zur Vorherhersage von sby_need
    prediction_calls.rename(columns={'pred': 'calls'}, inplace=True)
    X_calls = pd.concat([y_trainval, prediction_calls], axis=0)



    # Vorhersage n_sick
    y_name = 'n_sick'

    # Eigene Merkmalserzeugung und Auswahl
    X_trainval, X_test, y_trainval, y_test = featureextraction(df, y_name)

    # Erneutes Aufteilen der Trainingsdaten on Trainingsdaten und Validierungsdaten
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)


    # Vorhersage der Krankmeldungen mit LGBM
    metric_lgbm, prediction_sick, forecasterlgbm = lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test)

    # Ergebnisse vorbereiten zur Weiterverarbeitung zur Vorhersage von sby_need
    prediction_sick.rename(columns={'pred': 'n_sick'}, inplace=True)
    X_sick = pd.concat([y_trainval, prediction_sick], axis=0)

    # Direkter linearer Zusammenhang würde sich auch direkt berechnen lassen:
    # print(prediction_calls / 5)
    # print(df.loc[y_test.index, 'n_duty'])
    # print(prediction_sick)
    # print(prediction_calls / 5 - df.loc[y_test.index, 'n_duty'])

    # df_combined = df.loc[y_test.index, 'n_duty']
    # df_combined = pd.concat([df_combined,prediction_calls,prediction_sick])
    # df_combined['sby_need_pred']= df
    # sby_need_pred = prediction_calls / 5 - 1900 + prediction_sick
    # sby_need_pred = sby_need_pred.clip(lower=0)
    #
    # sby_need_pred_up = intervals_calls['upper_bound'] / 5 - 1900 + intervals_sick['upper_bound']
    # sby_need_pred_up = sby_need_pred_up.clip(lower=0)

    # Bestimmung sby_need mit besten Modell
    # Sby_need bestimmen mit den Vorhersagen von Calls und n_sick
    df = data_preparation('sickness_table.csv')
    y_name = 'sby_need'

    # Eigene Merkmalserzeugung und Auswahl
    X_trainval, X_test, y_trainval, y_test = featureextraction_sby(df, y_name, X_calls, X_sick)

    true_values = y_test.values

    # Erneutes Aufteilen der Trainingsdaten in Trainingsdaten und Validierungsdaten
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

    # Vorhersagen mit LGBM und Bestimmung der Perzentile mittels Bootstrapping
    metric, sby_need_pred, forecasterlgbm = lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test)
    metric, sby_need_pred_70 = probalistic_lgbm(forecasterlgbm, sby_need_pred, pd.concat([y_trainval, y_test]),
                                                pd.concat([X_trainval, X_test]), y_test, y_trainval.size,70)
    metric, sby_need_pred_80 = probalistic_lgbm(forecasterlgbm, sby_need_pred, pd.concat([y_trainval, y_test]),
                                                pd.concat([X_trainval, X_test]), y_test, y_trainval.size,80)
    metric, sby_need_pred_90 = probalistic_lgbm(forecasterlgbm, sby_need_pred, pd.concat([y_trainval, y_test]),
                                                pd.concat([X_trainval, X_test]), y_test, y_trainval.size,90)
    metric, sby_need_pred_95 = probalistic_lgbm(forecasterlgbm, sby_need_pred, pd.concat([y_trainval, y_test]),
                                                pd.concat([X_trainval, X_test]), y_test, y_trainval.size,95)

    # Überprüfung der Residuen
    residuals = y_test - sby_need_pred['pred']
    residuals_histo(residuals)
    residuals_scatter(residuals, sby_need_pred['pred'])
    feature_importance(forecasterlgbm)

    #Ersetzen aller negativer Werte in den Vorhersagen mit 0
    sby_need_pred = sby_need_pred.clip(lower=0)
    sby_need_pred_70 = sby_need_pred_70.clip(lower=0)
    sby_need_pred_80 = sby_need_pred_80.clip(lower=0)
    sby_need_pred_90 = sby_need_pred_90.clip(lower=0)
    sby_need_pred_95 = sby_need_pred_95.clip(lower=0)


    # Ausgabe der Metriken zu den einzelnen Modellen
    print('LGBM-Vorhersage:')
    metrics(sby_need_pred['pred'], y_test)

    # Momentane Verfahrensweise
    # Erstellen eines np.array mit dem Wert 90 und der Länge von y_test
    print("StatusQuo:")
    constant_values = np.full(len(y_test), 90)
    metrics(constant_values, y_test)

    print("Baseline:")
    baseline_predictions = baseline_seasonal_naive_prediction(y_trainval, y_test)
    metrics(baseline_predictions, y_test)

    print('Probabilisitische Vorhersage 70%:')
    metrics(sby_need_pred_70['upper_bound'], y_test)
    print('Probabilisitische Vorhersage 80%:')
    metrics(sby_need_pred_80['upper_bound'], y_test)
    print('Probabilisitische Vorhersage 90%:')
    metrics(sby_need_pred_90['upper_bound'], y_test)
    print('Probabilisitische Vorhersage 95%:')
    metrics(sby_need_pred_95['upper_bound'], y_test)


    #   Ergebnisplot der Vorhersage mit unterschiedlichen Perzentilen
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, true_values, label='True Values')
    plt.plot(y_test.index, constant_values, label='Status Quo')
    plt.plot(y_test.index, baseline_predictions, label='Baseline Prediction')
    # plt.plot(y_test.index, prediction_sby, label='SBY_need Prediction ')
    # plt.plot(y_test.index, intervals_sby['upper_bound'], label = 'sby_need Upper bound')
    plt.plot(y_test.index, sby_need_pred, label='LGBM Prediction')
    plt.plot(y_test.index, sby_need_pred_70['upper_bound'], label='70% Perzentil')
    plt.plot(y_test.index, sby_need_pred_80['upper_bound'], label='80% Perzentil')
    plt.plot(y_test.index, sby_need_pred_90['upper_bound'], label='90% Perzentil')
    plt.plot(y_test.index, sby_need_pred_95['upper_bound'], label='95% Perzentil')
    plt.legend()
    plt.title('Vergleich der Vorhersage mit dem unterschiedlichen Perzentilen')
    plt.show()




    #  Modellvergleiche
    # Rekursive Multi-Step-Verfahren zur Vorhersage von calls/n_sick
    df = data_preparation('sickness_table.csv')
    y_name = 'calls' # Für Modellvergleiche Krankmeldung hier n_sick eintragen

    # Eigene Merkmalserzeugung und Auswahl
    X_trainval, X_test, y_trainval, y_test = featureextraction(df, y_name)

    # Erneutes Aufteilen der Trainingsdaten on Trainingsdaten und Validierungsdaten
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)

    # print("Größe Trainingsdaten: {} Größe Validierungsdaten: {} Größe Testdaten: {}\n".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    # Vergleich der Modelle mit den jeweils besten Parametern
    print("Vergleich der Modelle mit den jeweils besten Parametern:")

    # Vorhersage mit Baselinemodel und Ausgabe der Metriken
    print("Baseline:")
    baseline_predictions = baseline_seasonal_naive_prediction(y_trainval, y_test)
    metrics(baseline_predictions, y_test)

    # Vorhersage mit LGBM und Ausgabe der Metriken
    print('LightGBM:')
    metric_lgbm, predictionlgbm, forecasterlgbm = lightgbm_prediction(X_trainval, X_test, y_trainval, y_train, y_test)
    print(metric_lgbm)
    metrics(predictionlgbm['pred'], y_test)
    #
    # Bildung Konfidenzintervall und Ausgabe der Metriken
    # print('Probabilistic:')
    # metric, intervals = probalistic_lgbm(forecasterlgbm, predictionlgbm, pd.concat([y_trainval, y_test]),
    #                                      pd.concat([X_trainval, X_test]), y_test, y_trainval.size)
    # metrics(intervals['pred'], y_test)
    # metrics(intervals['upper_bound'], y_test)
    #
    # Vorhersage mit XGBoost und Ausgabe der Metriken
    print('XGBoost:')
    metric_xgb, predictionxgb = xgboost_prediction(X_trainval, X_test, y_trainval, y_train, y_test)
    print(metric_xgb)
    metrics(predictionxgb['pred'], y_test)

    # Vorhersage mit RandomForest und Ausgabe der Metriken
    print('RFRegressor:')
    metric_rf, predictionrf = rfregressor_prediciton(X_trainval, X_test, y_trainval, y_train, y_test)
    print(metric_rf)
    metrics(predictionrf['pred'], y_test)

    # Vorhersage mit neuronalen Netzwerk(LSTM)
    print('LSTM-Predictions: ')
    metric, predictionlstm = lstm_prediction(pd.concat([X_train, y_train], axis=1), pd.concat([X_val, y_val], axis=1),
                                             y_trainval, pd.concat(
            [pd.concat([X_trainval, X_test]), pd.concat([y_trainval, y_test])], axis=1), 36, y_name)
    print(f' LSTM: {metric}')


    #   Ergebnisplot Rekursive Multistepvarianten
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='True Values')
    plt.plot(y_test.index, baseline_predictions, label='Baseline Prediction')
    plt.plot(y_test.index, predictionlgbm['pred'], label='LightGBM Prediction')
    # plt.plot(y_test.index, intervals['pred'], label='LightGBM Prediction')
    plt.plot(y_test.index, predictionxgb['pred'], label='XGBoost Prediction')
    #plt.plot(y_test.index, intervals['lower_bound'])
    #plt.plot(y_test.index, intervals['upper_bound'])
    #plt.fill_between(y_test.index, intervals['lower_bound'], intervals['upper_bound'], facecolor='blue', alpha=0.5)
    plt.plot(y_test.index, predictionrf, label='RF Regression Predictions')
    plt.plot(y_test.index, predictionlstm, label='NN Predictions')
    plt.legend()
    plt.title('Vergleich der Vorhersagen verschiedener Modelle - Rekursive Multistepvariante')
    plt.show()




    # Vergleich anderer Modellarten mit Lags in Featureauswahl
    # Eigene Merkmalserzeugung und Auswahl - Nochmal neu, da mit Lags andere größen bei Trainings- und Testdaten enstehen
    X_trainval, X_test, y_trainval, y_test = featureextraction_lags(df, y_name)

    # Erneutes Aufteilen der Trainingsdaten on Trainingsdaten und Validierungsdaten
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=False)


    print("Baseline:")
    baseline_predictions = baseline_seasonal_naive_prediction(y_trainval, y_test)
    metrics(baseline_predictions, y_test)

    # # Vorhersage mit Random Forest als direct Multistep Vorhersage und Ausgabe der Metriken
    print("RandomForestRegressor:")
    rf_predictions = random_forest_prediction(X_trainval, X_train, X_val, X_test, y_trainval, y_train, y_val)
    metrics(rf_predictions, y_test)

    # Vorhersage mit SVR und Ausgabe der Metriken
    print("Support Vector Regression:")
    svr_predictions = svr_prediction(X_trainval, X_train, X_val, X_test, y_trainval, y_train, y_val)
    metrics(svr_predictions, y_test)


    #   Ergebnisplot Direkte Multi-Step Varianten
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='True Values')
    plt.plot(y_test.index, baseline_predictions, label='Baseline Prediction')
    plt.plot(y_test.index, rf_predictions, label='RF Regression Predictions')
    plt.plot(y_test.index, svr_predictions, label='SVR Predictions')
    plt.legend()
    plt.title('Vergleich der Vorhersagen verschiedener Modelle')
    plt.show()

    # Nicht weiter verwendet - Ridge und Arima
    #  steps = 36
    #  data_train = df[:-steps]
    #  data_test  = df[-steps:]
    #
    #  # predictions_ridge =ridge_prediction(data_train)
    #  # metrics(predictions_ridge,data_test['sby_need'])
    #  #
    #  print("StatusQuo:")
    #  constant_values = np.full(len(data_test), 90)
    #  metrics(constant_values, data_test['sby_need'])
    #
    #  print("Baseline:")
    #  baseline_predictions = baseline_seasonal_naive_prediction(data_train['sby_need'], data_test['sby_need'])
    #  metrics(baseline_predictions, data_test['sby_need'])
    #
    #  print('ARIMA')
    #  arima_predictions = arima_prediction(data_train['sby_need'], data_test['sby_need'])
    #  metrics(arima_predictions['pred'],data_test['sby_need'])
    #
    #   #   Ergebnisplot Rekursive Multistepvarianten und LSTM(?)
    #  plt.figure(figsize=(14, 7))
    #  plt.plot(data_test.index, data_test['sby_need'].values, label='True Values')
    #  plt.plot(data_test.index, constant_values, label='Status Quo')
    #  plt.plot(data_test.index, baseline_predictions, label='Baseline Prediction')
    #  plt.plot(data_test.index, arima_predictions, label='Arima Prediction')
    # # plt.plot(data_test.index, predictions_ridge, label='RidgePrediction')
    #  plt.legend()
    #  plt.title('Vergleich der Vorhersagen verschiedener Modelle - Direct Multistepvariante')
    #  plt.show()

    # print("Sarima:")
    # arima_model(y_trainval)
    #  #


if __name__ == '__main__':
    main()
