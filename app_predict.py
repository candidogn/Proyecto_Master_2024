import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Cargar datos
@st.cache
def cargar_datos():
    # Asumiendo que 'datos_interacciones.csv' es el archivo con los datos necesarios
    data = pd.read_csv('datos_interacciones.csv')
    return data

# Entrenar modelo
def entrenar_modelo(data):
    X = data[['variable_1', 'variable_2', 'variable_3', 'variable_4', 'variable_5', 'variable_6', 'variable_7', 'variable_8']]
    y = data['objetivo']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo, scaler

# App
def main():
    st.title('Predicción de Interacciones')

    data = cargar_datos()
    modelo, scaler = entrenar_modelo(data)

    # Crear entradas para las variables
    inputs = []
    for i in range(1, 9):
        inp = st.number_input(f'Variable {i}', step=0.01)
        inputs.append(inp)

    if st.button('Predecir'):
        inputs_scaled = scaler.transform([inputs])
        resultado = modelo.predict(inputs_scaled)[0]
        if resultado == 0:
            st.success("Enruta la llamada al grupo")
        elif resultado == 1:
            st.success("Por favor confirma el motivo de la llamada")
        else:
            st.error("Predicción no válida")

if __name__ == '__main__':
    main()
