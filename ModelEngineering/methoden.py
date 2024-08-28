import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import holidays


def data_preparation(file_path):
    """
    Datenvorbereitung: Datumnsformat wird angepasst, Index gesetzt und Frequenz hinzugefügt
    Entfernt alle Daten ab dem 10. des letzten Monats aus dem DataFrame.
    Nicht benötigte Spalten (m_sby und dafted) werden gedropt

    Parameters:
    - df: DataFrame mit einem DatetimeIndex

    Returns:
    - Bereinigter DataFrame ohne Daten ab dem 10. des letzten Monats
    """

    # Laden und Vorbereitung der Daten ohne die erste Spalte
    df = pd.read_csv(file_path, usecols=lambda column: column not in ['Unnamed: 0'], parse_dates=True, dayfirst=True)

    # Umwandlung der Datums-Spalte in datetime-Format mit dayfirst=True
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # Angenommen, 'df' ist der DataFrame mit der Zeitreihe und 'date' ist der Index.
    # Wenn das Datum noch nicht der Index ist, tun Sie dies:
    df.set_index('date', inplace=True)

    # Sicherstellen, dass das Datum der Index ist
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('d')

    # Letzter Tag des Datensatzes
    last_day = df.index.max()

    # Erster Tag des letzten Monats
    first_day_last_month = (last_day - pd.offsets.MonthBegin(1)).replace(day=1)

    # Datum setzen, ab dem die Daten entfernt werden sollen
    cutoff_date = first_day_last_month + pd.Timedelta(days=15)

    # Filtern des DataFrames, um nur die Daten bis einschließlich zum 9. des letzten Monats zu behalten
    df = df[df.index < cutoff_date]

    # n_sby besitzt als Konstante keinen Wert zu Erstellung der Vorhersage. dafted ist linear abhängig von sby_need und bringt auch keinen Wert für eine Vorhersage. Beide werden also gedropt.
    df.drop(['n_sby', 'dafted'], inplace=True, axis=1)
    # print(df.info())
    return df


# Merkmalserzeugung und Merkmalsauswahl für rekursive Multi-step Modelle
def featureextraction(df, y_name):
    # Neue Features: Wochentag, Kalendarwoche und Monat
    df['weekday'] = df.index.weekday
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month

    # Ferien und Feiertage
    # Erstelle eine Datumsreihe für den Zeitraum von 2016-04-01 bis 2019-05-27
    date_range = pd.date_range(start='2016-04-01', end='2019-05-27')

    # Initialisiere den DataFrame
    df_holiday = pd.DataFrame(date_range, columns=['date'])
    # Feiertage für Deutschland (Berlin) abrufen
    de_holidays = holidays.Germany(state='BE')

    # Funktion zum Überprüfen, ob ein Datum ein Feiertag ist
    df_holiday['Feiertag'] = df_holiday['date'].apply(lambda x: x in de_holidays)

    # Manuell Ferienzeiten in Berlin hinzufügen (Zeiträume für Schulferien)
    ferien_zeitraeume = [
        ('2016-03-21', '2016-04-02'),  # Osterferien 2016
        ('2016-07-21', '2016-09-03'),  # Sommerferien 2016
        ('2016-10-17', '2016-10-28'),  # Herbstferien 2016
        ('2016-12-23', '2017-01-03'),  # Weihnachtsferien 2016
        ('2017-04-10', '2017-04-21'),  # Osterferien 2017
        ('2017-07-20', '2017-09-01'),  # Sommerferien 2017
        ('2017-10-23', '2017-11-04'),  # Herbstferien 2017
        ('2017-12-21', '2018-01-03'),  # Weihnachtsferien 2017
        ('2018-03-26', '2018-04-07'),  # Osterferien 2018
        ('2018-07-05', '2018-08-17'),  # Sommerferien 2018
        ('2018-10-22', '2018-11-03'),  # Herbstferien 2018
        ('2018-12-21', '2019-01-05'),  # Weihnachtsferien 2018
        ('2019-04-15', '2019-04-26'),  # Osterferien 2019
    ]

    # Funktion zum Überprüfen, ob ein Datum in den Ferien ist
    df_holiday['Ferien'] = df_holiday['date'].apply(
        lambda x: any(pd.Timestamp(start) <= x <= pd.Timestamp(end) for start, end in ferien_zeitraeume)
    )
    df_holiday.set_index('date', inplace=True)

    # Konvertiere die booleschen Spalten in 1 und 0
    df_holiday['Feiertag'] = df_holiday['Feiertag'].astype(int)
    df_holiday['Ferien'] = df_holiday['Ferien'].astype(int)

    # Entfernen von NaN-Werten
    df.dropna(inplace=True)

    # Zyklische Features einrichten
    month_encoded = cyclical_encoding(df['month'], cycle_length=12)
    week_of_year_encoded = cyclical_encoding(df['week_of_year'], cycle_length=52)
    week_day_encoded = cyclical_encoding(df['weekday'], cycle_length=7)

    # Exogene Variablen vorbereiten
    X = pd.DataFrame()
    X = pd.concat(
        [X, month_encoded, week_of_year_encoded, week_day_encoded, df_holiday['Feiertag'], df_holiday['Ferien']],
        axis=1)
    X.dropna(inplace=True)
    X.index = pd.to_datetime(X.index)
    X = X.asfreq('d')

    # Zielvariable
    y = df[y_name]

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# Merkmalserzeugung und Merkmalsauswahl für die Vorhersage von sby_need
def featureextraction_sby(df, y_name, X_calls, X_sick):
    # Neue Features: Wochentag, Kalendarwoche und Monat
    df['weekday'] = df.index.weekday
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month

    # Ferien und Feiertage
    # Erstelle eine Datumsreihe für den Zeitraum von 2016-04-01 bis 2019-05-27
    date_range = pd.date_range(start='2016-04-01', end='2019-05-27')

    # Initialisiere den DataFrame
    df_holiday = pd.DataFrame(date_range, columns=['date'])
    # Feiertage für Deutschland (Berlin) abrufen
    de_holidays = holidays.Germany(state='BE')

    # Funktion zum Überprüfen, ob ein Datum ein Feiertag ist
    df_holiday['Feiertag'] = df_holiday['date'].apply(lambda x: x in de_holidays)

    # Manuell Ferienzeiten in Berlin hinzufügen (Zeiträume für Schulferien)
    ferien_zeitraeume = [
        ('2016-03-21', '2016-04-02'),  # Osterferien 2016
        ('2016-07-21', '2016-09-03'),  # Sommerferien 2016
        ('2016-10-17', '2016-10-28'),  # Herbstferien 2016
        ('2016-12-23', '2017-01-03'),  # Weihnachtsferien 2016
        ('2017-04-10', '2017-04-21'),  # Osterferien 2017
        ('2017-07-20', '2017-09-01'),  # Sommerferien 2017
        ('2017-10-23', '2017-11-04'),  # Herbstferien 2017
        ('2017-12-21', '2018-01-03'),  # Weihnachtsferien 2017
        ('2018-03-26', '2018-04-07'),  # Osterferien 2018
        ('2018-07-05', '2018-08-17'),  # Sommerferien 2018
        ('2018-10-22', '2018-11-03'),  # Herbstferien 2018
        ('2018-12-21', '2019-01-05'),  # Weihnachtsferien 2018
        ('2019-04-15', '2019-04-26'),  # Osterferien 2019
    ]

    # Funktion zum Überprüfen, ob ein Datum in den Ferien ist
    df_holiday['Ferien'] = df_holiday['date'].apply(
        lambda x: any(pd.Timestamp(start) <= x <= pd.Timestamp(end) for start, end in ferien_zeitraeume)
    )
    df_holiday.set_index('date', inplace=True)

    # Konvertiere die booleschen Spalten in 1 und 0
    df_holiday['Feiertag'] = df_holiday['Feiertag'].astype(int)
    df_holiday['Ferien'] = df_holiday['Ferien'].astype(int)

    # Entfernen von NaN-Werten
    df.dropna(inplace=True)

    # Zyklische Features einrichten
    month_encoded = cyclical_encoding(df['month'], cycle_length=12)
    week_of_year_encoded = cyclical_encoding(df['week_of_year'], cycle_length=52)
    week_day_encoded = cyclical_encoding(df['weekday'], cycle_length=7)

    # Exogene Variablen mit X_calls und X_sick vorbereiten
    X = pd.DataFrame()
    X = pd.concat([X, X_calls, X_sick, month_encoded, week_of_year_encoded, week_day_encoded, df_holiday['Feiertag'],
                   df_holiday['Ferien']], axis=1)
    X.dropna(inplace=True)
    X.index = pd.to_datetime(X.index)
    X = X.asfreq('d')

    # Zielvariable
    y = df[y_name]

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# Merkmalserzeugung und Merkmalsauswahl für direkte Multi-step Modelle
def featureextraction_lags(df, y_name):
    # Neue Features
    df['weekday'] = df.index.weekday
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['rolling_mean_7'] = df[y_name].rolling(window=7).mean()
    df['calls_x_weekday'] = df['calls'] * df['weekday']

    #
    # Lags einrichten:
    for lag in range(35, 365):  # Lag um 1 Jahr
        df[f'sby_need_lag_{lag}'] = df['sby_need'].shift(lag)
        df[f'n_sick_lag_{lag}'] = df['n_sick'].shift(lag)
        df[f'calls_lag_{lag}'] = df['calls'].shift(lag)
        df[f'n_duty_lag_{lag}'] = df['n_duty'].shift(lag)
        df[f'rolling_mean_7_lag_{lag}'] = df['rolling_mean_7'].shift(lag)
        df[f'calls_x_weekday_lag_{lag}'] = df['calls_x_weekday'].shift(lag)

    # Ferien und Feiertage
    # Erstelle eine Datumsreihe für den Zeitraum von 2016-04-01 bis 2019-05-27
    date_range = pd.date_range(start='2016-04-01', end='2019-05-27')

    # Initialisiere den DataFrame
    df_holiday = pd.DataFrame(date_range, columns=['date'])
    # Feiertage für Deutschland (Berlin) abrufen
    de_holidays = holidays.Germany(state='BE')

    # Funktion zum Überprüfen, ob ein Datum ein Feiertag ist
    df_holiday['Feiertag'] = df_holiday['date'].apply(lambda x: x in de_holidays)

    # Manuell Ferienzeiten in Berlin hinzufügen (Zeiträume für Schulferien)
    ferien_zeitraeume = [
        ('2016-03-21', '2016-04-02'),  # Osterferien 2016
        ('2016-07-21', '2016-09-03'),  # Sommerferien 2016
        ('2016-10-17', '2016-10-28'),  # Herbstferien 2016
        ('2016-12-23', '2017-01-03'),  # Weihnachtsferien 2016
        ('2017-04-10', '2017-04-21'),  # Osterferien 2017
        ('2017-07-20', '2017-09-01'),  # Sommerferien 2017
        ('2017-10-23', '2017-11-04'),  # Herbstferien 2017
        ('2017-12-21', '2018-01-03'),  # Weihnachtsferien 2017
        ('2018-03-26', '2018-04-07'),  # Osterferien 2018
        ('2018-07-05', '2018-08-17'),  # Sommerferien 2018
        ('2018-10-22', '2018-11-03'),  # Herbstferien 2018
        ('2018-12-21', '2019-01-05'),  # Weihnachtsferien 2018
        ('2019-04-15', '2019-04-26'),  # Osterferien 2019
    ]

    # Funktion zum Überprüfen, ob ein Datum in den Ferien ist
    df_holiday['Ferien'] = df_holiday['date'].apply(
        lambda x: any(pd.Timestamp(start) <= x <= pd.Timestamp(end) for start, end in ferien_zeitraeume)
    )
    df_holiday.set_index('date', inplace=True)

    # Konvertiere die booleschen Spalten in 1 und 0
    df_holiday['Feiertag'] = df_holiday['Feiertag'].astype(int)
    df_holiday['Ferien'] = df_holiday['Ferien'].astype(int)

    # Entfernen von NaN-Werten
    df.dropna(inplace=True)

    # Zyklische Features einrichten
    month_encoded = cyclical_encoding(df['month'], cycle_length=12)
    week_of_year_encoded = cyclical_encoding(df['week_of_year'], cycle_length=52)
    week_day_encoded = cyclical_encoding(df['weekday'], cycle_length=7)

    # Exogene Variablen mit X_calls und X_sick vorbereiten
    X = pd.DataFrame()
    X = pd.concat(
        [X, month_encoded, week_of_year_encoded, week_day_encoded, df_holiday['Feiertag'], df_holiday['Ferien']],
        axis=1)

    X.dropna(inplace=True)
    X.index = pd.to_datetime(X.index)
    X = X.asfreq('d')

    # lags hinzufügen
    for lag in range(35, 365):
        X = pd.concat([X, df[[f'sby_need_lag_{lag}', f'n_sick_lag_{lag}', f'calls_lag_{lag}', f'n_duty_lag_{lag}',
                              f'rolling_mean_7_lag_{lag}', f'calls_x_weekday_lag_{lag}']]], axis=1)

    # Zielvariable
    y = df[y_name]

    # Aufteilen in Trainings- und Testdatensätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# Metriken
def metrics(predictions, y_test):
    # Anzahl der Tage an denen Nicht-Bereitschaftsfahrende aktiviert werden mussten
    understuffed_days = np.sum(predictions < y_test)
    print(f"Anzahl der Fälle, bei denen die Vorhersage kleiner ist als der tatsächliche Wert: {understuffed_days}")
    # Anteilig
    # Gesamte Anzahl der Tage
    total_days = len(y_test)
    # Anteil der Fälle, bei denen predictions < y_test
    percentage_less_than = (understuffed_days / total_days) * 100
    print(
        f"Anteil der Fälle, bei denen die Vorhersage kleiner ist als der tatsächliche Wert: {percentage_less_than:.2f}%")

    # Dafted: Anzahl der aktivierten Fahrer, die nicht in Bereitschaft sind
    dafted = np.where(y_test > predictions, y_test - predictions, 0)
    # Berechnung des durchschnittlichen Anteils von sby_need an n_sby
    # Das Abziehen von Dafted sorgt dafür, das maximal soviele Bereitschaftsfahrer gezählt werden, wie zur Verfügung standen.
    ratio = (sum(y_test - dafted) / sum(predictions)) * 100
    print(
        f"Anteil der tatsächlich benötigten Bereitschaftsfahrenden an den vorhergesagten Bereitschaftsfahrenden: {ratio:.2f}%")

    # Berechnung des MAE
    mae = mean_absolute_error(y_test, predictions)
    print(f'MAE: {mae}')
    # Berechnung RMSE
    rmse = root_mean_squared_error(y_test, predictions)
    print(f'RMSE: {rmse}')


# Zyklisches Codieren der Kalendardaten
def cyclical_encoding(data: pd.Series, cycle_length: int) -> pd.DataFrame:
    # Zyklisches Kodieren mit Sinus und Cosinus
    sin = np.sin(2 * np.pi * data / cycle_length)
    cos = np.cos(2 * np.pi * data / cycle_length)
    result = pd.DataFrame({
        f"{data.name}_sin": sin,
        f"{data.name}_cos": cos
    })

    return result


# Ausgabe Histogramm Residuen
def residuals_histo(residuals):
    # Erstelle das Histogramm
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='blue', edgecolor='black', alpha=0.7)

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title('Histogramm der Residuen')
    plt.xlabel('Residuenwerte')
    plt.ylabel('Häufigkeit')

    # Diagramm anzeigen
    plt.show()


# Ausgabe Scatterplot Residuen
def residuals_scatter(residuals, pred):
    # Erstellung des Punktediagramms
    plt.figure(figsize=(8, 6))
    plt.scatter(pred, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Horizontale Linie bei y=0

    # Diagrammbeschriftungen
    plt.title('Residuen vs. Vorhergesagte Werte')
    plt.xlabel('Vorhergesagte Werte (pred)')
    plt.ylabel('Residuen')

    # Plot anzeigen
    plt.show()


# Ausgabe Featureimportance als Barplot
def feature_importance(forecaster):
    # Feature Importances erhalten
    importance = forecaster.get_feature_importances()

    # Plotten der Feature Importances
    plt.figure(figsize=(10, 8))
    sns.barplot(y=importance['feature'], x=importance['importance'], orient='h')
    plt.title('Feature Importances')
    plt.show()


# Featureauswahl mit Random Forest für direkte multi-step Modelle
def random_forest_feature_selection(X_train, X_test, y_train, y_test):
    # Random Forest Modell trainieren
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Feature Importances erhalten
    importances = rf.feature_importances_
    feature_names = X_train.columns

    # Sortieren der Features nach ihrer Wichtigkeit
    sorted_indices = importances.argsort()[::-1]

    # Plotten der Feature Importances
    plt.figure(figsize=(10, 8))
    sns.barplot(y=feature_names[sorted_indices], x=importances[sorted_indices], orient='h')
    plt.title('Feature Importances (Random Forest)')
    plt.show()

    # Auswahl der wichtigsten Features (zum Beispiel die Top 20)
    important_features = feature_names[sorted_indices][:20]
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]

    return X_train_selected, X_test_selected, y_train, y_test
