from Functions import objective_func
import pandas as pd
import optuna
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Configurar el backend de matplotlib
#matplotlib.use('Agg')  # Usar backend no interactivo para evitar problemas con la visualización

# Carga de Datos
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

test = test[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
test.set_index(test.columns[0], inplace=True)

# Asegurarse de que los índices sean fechas
train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)

# Correr el estudio (optimizamos Sharpe ratio ajustado)
results = []

def wrapped_objective(trial):
    result = objective_func(trial, train, train=True)
    results.append(result)
    return result["adjusted_sharpe"] if not np.isnan(result["adjusted_sharpe"]) else -np.inf

study = optuna.create_study(direction="maximize")
study.optimize(wrapped_objective, n_trials=100)  # Reducido a 100 trials para optimizar recursos

# Mostrar resultados finales
print("\n🔍 Mejor Sharpe (entrenamiento):", round(study.best_value, 4))
print("⚙  Mejores parámetros:", study.best_params)

# Extraer métricas del mejor resultado (entrenamiento)
best_index = max(
    range(len(results)),
    key=lambda i: results[i]["adjusted_sharpe"] if not np.isnan(results[i]["adjusted_sharpe"]) else -np.inf
)
best_result = results[best_index]

print("\n📊 Métricas del mejor resultado (entrenamiento):")
for k, v in best_result.items():
    if k != "return":  # Excluir la lista de retornos
        if isinstance(v, float):
            print(f"{k.capitalize():<25}: {round(v, 4)}")
        else:
            print(f"{k.capitalize():<25}: {v}")

# Evaluar en datos de prueba
best_params = study.best_params
test_result = objective_func(optuna.trial.FixedTrial(best_params), test, train=False)

print("\n📊 Métricas en datos de prueba:")
for k, v in test_result.items():
    if k != "return":  # Excluir la lista de retornos
        if isinstance(v, float):
            print(f"{k.capitalize():<25}: {round(v, 4)}")
        else:
            print(f"{k.capitalize():<25}: {v}")

# Visualizar el valor del portafolio (entrenamiento y prueba)
try:
    plt.figure(figsize=(12, 6))
    # Verificar que las longitudes coincidan
    train_len = min(len(train.index), len(best_result["return"]))
    test_len = min(len(test.index), len(test_result["return"]))

    # Depuración: Mostrar las longitudes
    print(f"\n📏 Longitud de train.index: {len(train.index)}, best_result['return']: {len(best_result['return'])}, train_len: {train_len}")
    print(f"📏 Longitud de test.index: {len(test.index)}, test_result['return']: {len(test_result['return'])}, test_len: {test_len}")

    # Verificar que las longitudes no sean 0
    if train_len == 0 or test_len == 0:
        raise ValueError("Una de las longitudes es 0, no se puede graficar.")

    plt.plot(train.index[:train_len], best_result["return"][:train_len], label='Entrenamiento', color='blue')
    plt.plot(test.index[:test_len], test_result["return"][:test_len], label='Prueba', color='orange')
    plt.title('Evolución del Valor del Portafolio')
    plt.xlabel('Fecha')
    plt.ylabel('Valor ($)')
    plt.legend()
    plt.grid()
    plt.show()
    # Guardar el gráfico como archivo
    plt.savefig('portfolio_value.png')
    print("\n📈 Gráfico guardado como 'portfolio_value.png'")

except Exception as e:
    print(f"\n❌ Error al generar el gráfico: {e}")
finally:
    plt.close()  # Cerrar la figura para liberar memoria