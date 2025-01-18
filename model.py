import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Wczytanie danych
data = pd.read_csv('USA Housing Dataset.csv')
d = pd.read_csv('USA Housing Dataset.csv')


# Usuwanie kolumny 'country' oraz kodowanie kolumn kategorycznych
data.drop(['country','waterfront','view','condition', 'sqft_above','sqft_basement'], axis='columns', inplace=True)
data = pd.get_dummies(data, columns=['city', 'statezip'], drop_first=True)

# Konwersja kolumny 'date' na format datetime i generowanie nowych cech
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek
data['quarter'] = data['date'].dt.quarter

# Usuwanie kolumny 'date'
data = data.drop(columns=['date'])

scaler = MinMaxScaler()
num_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'year', 'month', 'day', 'day_of_week', 'quarter']
data[num_columns] = scaler.fit_transform(data[num_columns])

# Przygotowanie danych do modelu
X = data.drop(columns=['price', 'street'])  # Wszystkie kolumny poza 'price' i 'street'
y = data['price']  # Cena jako zmienna zależna

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu regresji liniowej
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
y_pred = model_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------------------------
# Aplikacja Streamlit
# -----------------------------------------------
# Nagłówek
st.title("Predykcja cen nieruchomości")
st.write("Model regresji liniowej do przewidywania cen domów na podstawie wybranych cech.")

# Wyświetlenie oceny modelu
st.subheader("Ocena modelu:")
st.write(f"Średni błąd kwadratowy (MSE): {mse:.4f}")
st.write(f"Współczynnik determinacji (R2): {r2:.4f}")

# Wprowadzenie danych wejściowych
st.subheader("Wprowadź dane nieruchomości:")
bedrooms = st.number_input("Liczba sypialni", min_value=0.0, step=1.0)
bathrooms = st.number_input("Liczba łazienek", min_value=0.0, step=0.5)
sqft_living = st.number_input("Powierzchnia mieszkalna (sqft)", min_value=0.0, step=10.0)
sqft_lot = st.number_input("Powierzchnia działki (sqft)", min_value=0.0, step=10.0)
floors = st.number_input("Liczba pięter", min_value=0.0, step=0.5)
year = st.number_input("Rok budowy", min_value=1900, step=1, value=2000)
month = st.number_input("Miesiąc zakupu (1-12)", min_value=1, max_value=12, step=1)
day = st.number_input("Dzień zakupu (1-31)", min_value=1, max_value=31, step=1)
day_of_week = st.number_input("Dzień tygodnia (0=Poniedziałek, 6=Niedziela)", min_value=0, max_value=6, step=1)
quarter = st.number_input("Kwartał roku (1-4)", min_value=1, max_value=4, step=1)


# Wybór miasta z listy unikalnych wartości
city = st.selectbox("Wybierz miasto", options=d['city'].unique())

# Przygotowanie danych wejściowych
if st.button("Oblicz cenę"):
    # Tworzenie wiersza danych wejściowych
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, year, month, day, day_of_week, quarter]], 
                              columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'year', 'month', 'day', 'day_of_week', 'quarter'])
    
    # Kodowanie miasta
    city_encoded = pd.get_dummies([city], prefix='city')
    city_encoded = city_encoded.drop(columns=city_encoded.columns[0])  # Drop the first city column to avoid dummy variable trap
    input_data = pd.concat([input_data, city_encoded], axis=1)
    
    # Dopasowanie do wymiarów danych
    missing_columns = set(X.columns) - set(input_data.columns)
    for col in missing_columns:
        input_data[col] = 0
    input_data = input_data[X.columns]  # Zachowanie właściwej kolejności kolumn

    # Skalowanie danych wejściowy
    input_data[num_columns] = scaler.transform(input_data[num_columns])

    # Predykcja
    prediction = model_lr.predict(input_data)
    st.subheader("Przewidywana cena:")
    st.write(f"Przewidywana cena: ${prediction[0]:,.2f}")
