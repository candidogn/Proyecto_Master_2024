import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
@st.cache
def load_data():
    datos = pd.read_csv('datos_interacciones.csv')
    return datos

datos = load_data()

# Dividir los datos en entrenamiento y prueba
X = datos[['variable_1', 'variable_2', 'variable_3', 'variable_4', 'variable_5', 'variable_6', 'variable_7', 'variable_8']]
y = datos['valor_objetivo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Crear la aplicación Streamlit
def main():
    st.title('Predicción de Regresión Múltiple')

    # Entradas del usuario para las variables
    v1 = st.number_input('Inserte variable_1')
    v2 = st.number_input('Inserte variable_2')
    v3 = st.number_input('Inserte variable_3')
    v4 = st.number_input('Inserte variable_4')
    v5 = st.number_input('Inserte variable_5')
    v6 = st.number_input('Inserte variable_6')
    v7 = st.number_input('Inserte variable_7')
    v8 = st.number_input('Inserte variable_8')

    if st.button('Predecir'):
        # Realizar la predicción
        resultado = modelo.predict([[v1, v2, v3, v4, v5, v6, v7, v8]])
        st.write(f'El valor objetivo predicho es: {resultado[0]}')

if __name__ == '__main__':
    main()
