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
data = pd.read_csv('Cleaned.csv')
d = pd.read_csv('USA Housing Dataset.csv')

# Przygotowanie danych do modelu
X = data.drop(columns=['price', 'street'])  # Wszystkie kolumny poza 'price' i 'street'
y = data['price']  # Cena jako zmienna zależna

# Skalowanie danych
num_columns = X.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
X[num_columns] = scaler.fit_transform(X[num_columns])

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

# Wybór miasta z listy unikalnych wartości
unique_cities = d['city'].unique()
city = st.selectbox("Wybierz miasto", options=unique_cities)

# Przygotowanie danych wejściowych
if st.button("Oblicz cenę"):
    # Tworzenie wiersza danych wejściowych
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, year, month]], 
                              columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'year', 'month'])
    
    # Skalowanie danych wejściowych na podstawie skalera użytego do danych treningowych
    input_data[num_columns] = scaler.transform(input_data[num_columns])

    # Kodowanie miasta (one-hot encoding)
    city_encoded = pd.get_dummies([city], prefix='city')
    for col in [col for col in X.columns if col.startswith('city_')]:
        city_encoded[col] = 0
    city_encoded[f'city_{city}'] = 1
    
    # Łączenie danych wejściowych z zakodowanymi miastami
    input_data = pd.concat([input_data, city_encoded], axis=1)

    # Dopasowanie do wymiarów danych (dodanie brakujących kolumn jako 0)
    missing_columns = set(X.columns) - set(input_data.columns)
    for col in missing_columns:
        input_data[col] = 0
    input_data = input_data[X.columns]  # Zachowanie właściwej kolejności kolumn

    # Predykcja
    prediction = model_lr.predict(input_data)
    st.subheader("Przewidywana cena:")
    st.write(f"Przewidywana cena: ${prediction[0]:,.2f}")
