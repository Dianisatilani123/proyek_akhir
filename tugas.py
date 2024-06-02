import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load dataset
dataset = pd.read_csv("dataset.csv")

# Display dataset
st.write("Dataset:")
st.write(dataset.head())

# Assume that the last column is the target variable (y)
X = dataset.iloc[:, :-1]  # features
y = dataset.iloc[:, -1]  # target variable

# Standarisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membuat model latih menggunakan Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Membuat model evaluasi untuk uji akurasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write("Mean Squared Error:", mse)

# Membuat model aplikasi untuk prediksi
def predict_proporsi(kabupaten, tahun):
    X_new = np.array([[kabupaten, tahun]])
    X_new_scaled = scaler.transform(X_new)
    y_pred = model.predict(X_new_scaled)
    return y_pred[0]

# Deploy aplikasi AI ke web online menggunakan Streamlit
st.title("Cerah Masa Depan")
st.write("Aplikasi ini membantu remaja membuat keputusan yang lebih baik tentang masa depan mereka.")

# Input form untuk kabupaten dan tahun
kabupaten = st.selectbox("Pilih Kabupaten:", dataset["Kabupaten/Kota"].unique())
tahun = st.selectbox("Pilih Tahun:", [2021, 2022, 2023])

# Button untuk prediksi
if st.button("Prediksi"):
    proporsi = predict_proporsi(kabupaten, tahun)
    st.write("Proporsi perempuan pernah kawin usia 15-49 tahun yang melahirkan anak lahir hidup yang pertama kali berumur kurang dari 20 tahun (MPK20) di", kabupaten, "pada tahun", tahun, "adalah", proporsi)
