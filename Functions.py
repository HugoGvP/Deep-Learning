from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Métricas ---
def calculate_returns(portafolio_value):
    return np.diff(portafolio_value) / portafolio_value[:-1]


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=19656):
    if returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=19656):
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    sortino = np.mean(excess_returns) / np.std(downside_returns)
    return sortino * np.sqrt(periods_per_year)


def calculate_calmar_ratio(portafolio_value, periods_per_year=19656):
    returns = calculate_returns(portafolio_value)
    peak = np.maximum.accumulate(portafolio_value)
    drawdown = (portafolio_value - peak) / peak
    max_drawdown = np.min(drawdown)
    annual_return = np.mean(returns) * periods_per_year
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


def calculate_win_loss_ratio(transacciones, data):
    wins = 0
    losses = 0
    for tx in transacciones:
        entry_price = tx['Precio']
        exit_index = tx.get('Exit')
        if exit_index is None or exit_index not in data.index:
            continue
        exit_price = data.loc[exit_index, 'Close']
        pnl = (exit_price - entry_price) if tx['Tipo'] else (entry_price - exit_price)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
    total = wins + losses
    return wins / total if total > 0 else 0


# --- Análisis de transacciones ---
def analyze_transactions(transacciones, data):
    if not transacciones:
        return {"num_trades": 0, "avg_win": 0, "avg_loss": 0}

    wins = []
    losses = []
    for tx in transacciones:
        entry_price = tx['Precio']
        exit_index = tx.get('Exit')
        if exit_index is None or exit_index not in data.index:
            continue
        exit_price = data.loc[exit_index, 'Close']
        pnl = (exit_price - entry_price) if tx['Tipo'] else (entry_price - exit_price)
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(pnl)

    num_trades = len(wins) + len(losses)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    return {
        "num_trades": num_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }


# --- Función principal ---
def objective_func(trial, data, train=True):
    data = data.copy()
    # Verificar columnas necesarias
    required_columns = ['High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan las siguientes columnas en el DataFrame: {missing_columns}")

    # Parámetros optimizados
    rsi_window = trial.suggest_int('rsi_window', 5, 20)
    macd_fast = trial.suggest_int('macd_fast', 5, 15)
    macd_slow = trial.suggest_int('macd_slow', 20, 40)
    macd_signal = trial.suggest_int('macd_signal', 5, 15)
    stoch_window = trial.suggest_int('stoch_window', 5, 20)
    stoch_smooth = trial.suggest_int('stoch_smooth', 3, 10)
    take_profit = trial.suggest_float('take_profit', 0.05, 0.15)
    stop_loss_ratio = trial.suggest_float('stop_loss_ratio', 0.1, 0.3)  # Reducido para un stop_loss más ajustado
    stop_loss = take_profit * stop_loss_ratio
    trailing_stop = trial.suggest_float('trailing_stop', 0.02, 0.05)  # Trailing stop entre 2% y 5%
    n_shares = trial.suggest_categorical('n_shares', [4000, 4500, 5000, 5500, 6000, 6500, 7000])
    cooldown_period = trial.suggest_int('cooldown_period', 0, 3)
    volume_threshold = trial.suggest_float('volume_threshold', 0.5, 2.0)  # Umbral de volumen relativo al promedio

    # Indicadores técnicos
    data['RSI'] = RSIIndicator(close=data['Close'], window=rsi_window).rsi()
    macd = MACD(close=data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    data['MACD'] = macd.macd()
    data['MACD_Signal_Line'] = macd.macd_signal()
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=stoch_window,
                                 smooth_window=stoch_smooth)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    # Filtro de tendencia (SMA 200)
    data['SMA_200'] = SMAIndicator(close=data['Close'], window=200).sma_indicator()
    # Filtro de volumen
    avg_volume = data['Volume'].mean()
    data['Volume_Filter'] = (data['Volume'] > avg_volume * volume_threshold).astype(int)
    data = data.dropna()

    # Generar señales
    data['RSI_Signal'] = 0
    data.loc[(data['RSI'] < 45) & (data['RSI'].shift(1) >= 45), 'RSI_Signal'] = 1  # Relajado
    data.loc[(data['RSI'] > 55) & (data['RSI'].shift(1) <= 55), 'RSI_Signal'] = -1  # Relajado

    data['MACD_Signal'] = 0
    data.loc[(data['MACD'] > data['MACD_Signal_Line']) & (
                data['MACD'].shift(1) <= data['MACD_Signal_Line'].shift(1)), 'MACD_Signal'] = 1
    data.loc[(data['MACD'] < data['MACD_Signal_Line']) & (
                data['MACD'].shift(1) >= data['MACD_Signal_Line'].shift(1)), 'MACD_Signal'] = -1

    data['Stoch_Signal'] = 0
    data.loc[(data['Stoch_K'] > data['Stoch_D']) & (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)) & (
                data['Stoch_K'] < 35), 'Stoch_Signal'] = 1  # Relajado
    data.loc[(data['Stoch_K'] < data['Stoch_D']) & (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1)) & (
                data['Stoch_K'] > 65), 'Stoch_Signal'] = -1  # Relajado

    # Filtro de tendencia
    data['Trend'] = 0
    data.loc[data['Close'] > data['SMA_200'], 'Trend'] = 1  # Tendencia alcista
    data.loc[data['Close'] < data['SMA_200'], 'Trend'] = -1  # Tendencia bajista

    # Señal combinada
    signal_sum = data[['RSI_Signal', 'MACD_Signal', 'Stoch_Signal']].sum(axis=1)
    data['Cash_medio'] = 0
    data.loc[(signal_sum >= 1) & (data['Trend'] == 1) & (data['Volume_Filter'] == 1), 'Cash_medio'] = 1  # Compra
    data.loc[(signal_sum <= -1) & (data['Trend'] == -1) & (data['Volume_Filter'] == 1), 'Cash_medio'] = -1  # Venta

    capital = 1_000_000
    comision = 0.00125
    slippage = 0.0015
    transacciones = []
    portafolio_value = []
    active_position = {}
    last_trade_index = -cooldown_period - 1
    highest_price = None  # Para trailing stop (long)
    lowest_price = None  # Para trailing stop (short)

    for i in data.index:
        precio_actual = data.loc[i, 'Close']

        # Entrada de posición
        if not active_position:
            if (data.index.get_loc(i) - last_trade_index) > cooldown_period:
                if data.loc[i, 'Cash_medio'] == 1:
                    active_position = {
                        'Date': i,
                        'Precio': precio_actual,
                        'Tipo': True,  # long
                        'Shares': n_shares,
                        'Inversion': precio_actual * n_shares * (1 + comision + slippage)
                    }
                    capital -= active_position['Inversion']
                    highest_price = precio_actual
                elif data.loc[i, 'Cash_medio'] == -1:
                    active_position = {
                        'Date': i,
                        'Precio': precio_actual,
                        'Tipo': False,  # short
                        'Shares': n_shares,
                        'Inversion': precio_actual * n_shares * (1 + comision + slippage)
                    }
                    capital -= active_position['Inversion']
                    lowest_price = precio_actual

        # Gestión de salida
        if active_position:
            precio_entrada = active_position['Precio']
            tipo = active_position['Tipo']
            shares = active_position['Shares']

            if tipo:  # Long
                # Actualizar el precio más alto para trailing stop
                highest_price = max(highest_price, precio_actual)
                trailing_stop_price = highest_price * (1 - trailing_stop)
                target = precio_entrada * (1 + take_profit)
                stop = precio_entrada * (1 - stop_loss)
                if precio_actual >= target or precio_actual <= stop or precio_actual <= trailing_stop_price:
                    salida = precio_actual * shares * (1 - comision - slippage)
                    capital += salida
                    pnl = salida - active_position['Inversion']
                    active_position.update({'Exit': i, 'PnL': pnl})
                    transacciones.append(active_position)
                    last_trade_index = data.index.get_loc(i)
                    active_position = {}
                    highest_price = None
            else:  # Short
                # Actualizar el precio más bajo para trailing stop
                lowest_price = min(lowest_price, precio_actual)
                trailing_stop_price = lowest_price * (1 + trailing_stop)
                target = precio_entrada * (1 - take_profit)
                stop = precio_entrada * (1 + stop_loss)
                if precio_actual <= target or precio_actual >= stop or precio_actual >= trailing_stop_price:
                    salida = (precio_entrada - precio_actual) * shares * (1 - comision - slippage)
                    capital += active_position['Inversion'] + salida
                    pnl = salida
                    active_position.update({'Exit': i, 'PnL': pnl})
                    transacciones.append(active_position)
                    last_trade_index = data.index.get_loc(i)
                    active_position = {}
                    lowest_price = None

        # Valor del portafolio
        if active_position:
            if active_position['Tipo']:  # long
                flotante = (precio_actual - active_position['Precio']) * n_shares
            else:  # short
                flotante = (active_position['Precio'] - precio_actual) * n_shares
            portafolio_value.append(capital + active_position['Inversion'] + flotante)
        else:
            portafolio_value.append(capital)

    if len(portafolio_value) < 2:
        return {
            "return": [capital],
            "sharpe": 0,
            "sortino": 0,
            "calmar": 0,
            "win_loss": 0,
            "final_portfolio_value": capital,
            "num_trades": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "adjusted_sharpe": 0
        }

    returns = calculate_returns(np.array(portafolio_value))
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(np.array(portafolio_value))
    win_loss = calculate_win_loss_ratio(transacciones, data)
    final_value = portafolio_value[-1]
    trade_stats = analyze_transactions(transacciones, data)

    # Penalizar Win/Loss bajo (penalización más fuerte)
    penalty = max(0, 0.4 - win_loss) * 15  # Aumentada la penalización
    adjusted_sharpe = sharpe - penalty

    return {
        "return": portafolio_value,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_loss": win_loss,
        "final_portfolio_value": final_value,
        "num_trades": trade_stats["num_trades"],
        "avg_win": trade_stats["avg_win"],
        "avg_loss": trade_stats["avg_loss"],
        "adjusted_sharpe": adjusted_sharpe
    }