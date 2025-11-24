import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA CARTERA ---
PORTFOLIO_CONFIG = {
    'DJMC.AS': {'name': 'iShares EURO STOXX Mid', 'weight': 0.30, 'type': 'Renta Variable Europa'},
    'EXH9.DE': {'name': 'iShares STOXX Eu 600 Util', 'weight': 0.129, 'type': 'Sectorial Utilities'},
    'ISPA.DE': {'name': 'iShares Global Select Div 100', 'weight': 0.171, 'type': 'Global Dividend'},
    'IUSM.DE': {'name': 'iShares USD Treasury 7-10yr', 'weight': 0.30, 'type': 'Bonos Gobierno'},
    'SYBJ.DE': {'name': 'SPDR Bloomberg Euro HY Bond', 'weight': 0.10, 'type': 'Bonos High Yield'}
}

RISK_FREE_RATE = 0.03  # 3% tasa libre de riesgo aproximada (para Sharpe)

# --- FUNCIONES DE CÁLCULO ---
def get_data(tickers, start_date):
    data = yf.download(tickers, start=start_date)['Adj Close']
    return data

def calculate_metrics(series):
    # Retornos diarios
    returns = series.pct_change().dropna()
    
    # CAGR
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max Drawdown
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Ratio Sharpe (Anualizado)
    excess_returns = returns - (RISK_FREE_RATE / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    return cagr, max_dd, sharpe, drawdown

# --- INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(page_title="Cartera Permanente Modificada", layout="wide", page_icon="📈")

st.title("📊 Dashboard de Control de Cartera")
st.markdown(f"**Estrategia:** Bandas ±10% | **Capital Inicial:** Configurable")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    initial_capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    start_date = st.date_input("Fecha de Inicio", value=datetime.now() - timedelta(days=365*2))
    
    st.divider()
    st.write("Asignación Teórica:")
    df_alloc = pd.DataFrame.from_dict(PORTFOLIO_CONFIG, orient='index')
    st.dataframe(df_alloc[['weight', 'name']], use_container_width=True)

# --- LÓGICA PRINCIPAL ---
if st.button('Actualizar Datos y Calcular'):
    with st.spinner('Descargando datos de mercado...'):
        tickers = list(PORTFOLIO_CONFIG.keys())
        df_prices = get_data(tickers, start_date)
        
        # 1. Calcular valor teórico de la cartera hoy (Buy & Hold desde inicio)
        # Asumimos compra en fecha de inicio
        initial_prices = df_prices.iloc[0]
        current_prices = df_prices.iloc[-1]
        
        # Calcular acciones compradas
        shares = {}
        current_values = {}
        total_value = 0
        
        for ticker, info in PORTFOLIO_CONFIG.items():
            allocated_cash = initial_capital * info['weight']
            num_shares = allocated_cash / initial_prices[ticker]
            shares[ticker] = num_shares
            val = num_shares * current_prices[ticker]
            current_values[ticker] = val
            total_value += val
            
        # 2. Crear serie histórica de la cartera
        portfolio_history = pd.DataFrame()
        for ticker in tickers:
            portfolio_history[ticker] = df_prices[ticker] * shares[ticker]
        
        portfolio_history['Total'] = portfolio_history.sum(axis=1)
        
        # 3. Calcular Métricas
        cagr, max_dd, sharpe, dd_series = calculate_metrics(portfolio_history['Total'])
        
        # --- VISUALIZACIÓN ---
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valor Actual", f"{total_value:,.2f} €", delta=f"{total_value - initial_capital:,.2f} €")
        col2.metric("CAGR", f"{cagr*100:.2f}%")
        col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
        col4.metric("Ratio Sharpe", f"{sharpe:.3f}")
        
        # Gráfico Principal
        st.subheader("Evolución del Patrimonio")
        fig = px.area(portfolio_history, x=portfolio_history.index, y='Total', title="Crecimiento de la Inversión")
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de Drawdown
        with st.expander("Ver Gráfico de Drawdown (Caídas desde máximos)"):
            fig_dd = px.line(dd_series, title="Drawdown Submarino")
            fig_dd.add_hrect(y0=-1, y1=-0.20, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Zona de Riesgo")
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- SECCIÓN DE REBALANCEO ---
        st.header("⚖️ Estado del Rebalanceo (Bandas ±10%)")
        
        rebal_data = []
        alert_triggered = False
        
        for ticker, info in PORTFOLIO_CONFIG.items():
            target_w = info['weight']
            current_w = current_values[ticker] / total_value
            diff = current_w - target_w
            
            # Lógica de Bandas: Si la desviación relativa es > 10% de su peso objetivo
            # OJO: La estrategia "Bandas 10%" suele significar desviación relativa.
            # Ejemplo: Si peso es 30%, banda es 27% - 33% (si es relativa al peso)
            # O absoluta: 20% - 40%. Asumiremos RELATIVA al peso objetivo (más estricto).
            
            upper_limit = target_w * 1.10
            lower_limit = target_w * 0.90
            
            status = "✅ OK"
            action = "Mantener"
            
            if current_w > upper_limit:
                status = "🔴 VENDER"
                action = f"Vender {current_values[ticker] - (total_value * target_w):.2f} €"
                alert_triggered = True
            elif current_w < lower_limit:
                status = "🔵 COMPRAR"
                action = f"Comprar {(total_value * target_w) - current_values[ticker]:.2f} €"
                alert_triggered = True
                
            rebal_data.append({
                "Ticker": ticker,
                "Nombre": info['name'],
                "Peso Objetivo": f"{target_w*100:.1f}%",
                "Peso Actual": f"{current_w*100:.2f}%",
                "Estado": status,
                "Acción Necesaria": action
            })
            
        df_rebal = pd.DataFrame(rebal_data)
        
        # Colorear la tabla
        def color_status(val):
            color = 'red' if 'VENDER' in val else 'blue' if 'COMPRAR' in val else 'green'
            return f'color: {color}; font-weight: bold'

        st.dataframe(df_rebal.style.applymap(color_status, subset=['Estado']), use_container_width=True)
        
        if alert_triggered:
            st.warning("⚠️ ¡ATENCIÓN! Algunos activos han superado las bandas de tolerancia del 10%. Se recomienda rebalancear.")
        else:
            st.success("La cartera está equilibrada dentro de las bandas establecidas.")
