from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np

# Valores para los parámetros
S_values = np.linspace(50, 150, 100)
T_values = np.linspace(0.1, 2, 100)
r_values = np.linspace(0.01, 0.1, 100)
sigma_values = np.linspace(0.1, 0.5, 100)

def black_scholes_fd(S, K, T, r, sigma, option_type='call', M=100, N=100):

    dt = T / N
    ds = 2 * S / M

    # Crear malla de precios del activo subyacente
    S_values = np.arange(0, 2 * S + ds, ds)

    # Inicializar matriz de precios de opciones
    option_prices = np.zeros((N + 1, len(S_values)))

    # Condiciones de contorno
    if option_type == 'call':
        option_prices[:, :] = np.maximum(S_values - K, 0)
        option_prices[-1, :] = np.maximum(S_values - K, 0)
        option_prices[:, 0] = 0
        option_prices[:, -1] = S_values[-1] - K * np.exp(-r * (N - np.arange(N + 1)) * dt)
    elif option_type == 'put':
        option_prices[:, :] = np.maximum(K - S_values, 0)
        option_prices[-1, :] = np.maximum(K - S_values, 0)
        option_prices[:, 0] = K * np.exp(-r * (N - np.arange(N + 1)) * dt)
        option_prices[:, -1] = 0

    # Calcular precios de opciones
    for j in range(N - 1, -1, -1):
        for i in range(1, len(S_values) - 1):
            a = 0.5 * sigma**2 * i**2 * dt
            b = 0.5 * r * i * dt
            c = -r
            option_prices[j, i] = a * option_prices[j + 1, i - 1] + (1 - a - dt * c) * option_prices[j + 1, i] + b * option_prices[j + 1, i + 1]

    # Interpolar para obtener el precio de la opción para S=S0
    option_price = np.interp(S, S_values, option_prices[0, :])

    return option_price

#-----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generar_graficas', methods=['POST'])
def generar_graficas():
    SO = float(request.form['SO'])
    K = float(request.form['k'])
    T = float(request.form['T'])
    r = float(request.form['r'])
    sigma = float(request.form['sigma'])

    # Generar las gráficas aquí usando matplotlib y los valores ingresados
    call_prices_S = [black_scholes_fd(SO, K, T, r, sigma) for SO in S_values]
    call_prices_T = [black_scholes_fd(SO, K, T, r, sigma) for T in T_values]
    call_prices_r = [black_scholes_fd(SO, K, T, r, sigma) for r in r_values]
    call_prices_sigma = [black_scholes_fd(SO, K, T, r, sigma) for sigma in sigma_values]

    # Graficar precios de opciones en función de los parámetros
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(S_values, call_prices_S)
    plt.xlabel('Precio del activo subyacente (S)')
    plt.ylabel('Precio de la opción de compra')
    plt.title('Precio de la opción de compra vs Precio del activo subyacente')

    plt.subplot(2, 2, 2)
    plt.plot(T_values, call_prices_T)
    plt.xlabel('Tiempo hasta la expiración (T)')
    plt.ylabel('Precio de la opción de compra')
    plt.title('Precio de la opción de compra vs Tiempo hasta la expiración')

    plt.subplot(2, 2, 3)
    plt.plot(r_values, call_prices_r)
    plt.xlabel('Tasa de interés (r)')
    plt.ylabel('Precio de la opción de compra')
    plt.title('Precio de la opción de compra vs Tasa de interés')

    plt.subplot(2, 2, 4)
    plt.plot(sigma_values, call_prices_sigma)
    plt.xlabel('Volatilidad (sigma)')
    plt.ylabel('Precio de la opción de compra')
    plt.title('Precio de la opción de compra vs Volatilidad')

    plt.tight_layout()  # Ajustar el diseño de las sub-gráficas
    plt.savefig('static/graficas.png')  # Guardar la gráfica como imagen en la carpeta 'static'
    plt.close()

    return render_template('graficas.html')

if __name__ == '__main__':
    app.run(debug=True)
