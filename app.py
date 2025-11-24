import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. CONFIGURACIÓN VISUAL (ESTILO PRO) ---
st.set_page_config(page_title="Cartera Permanente", layout="wide", page_icon="📈")

# CSS Personalizado: Fondo Gris Azulado (Slate) en lugar de Negro Puro
st.markdown("""
<style>
    /* Fondo general más suave (Gris Oscuro Azulado) */
    .stApp {
        background-color: #1e2433;
    }
    
    /* Textos generales */
    h1, h2, h3, p, div, span {
        color: #e2e8f0 !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 15px;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #cbd5e0 !important; /* Gris claro para etiquetas */
        font-size: 1rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important; /* Blanco puro para números */
        font-size: 1.8rem !important;
    }
    
    /* Tablas */
    .dataframe {
        font-size: 1.1rem !important; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DATOS ---
PORTFOLIO_CONFIG = {
    'DJMC.AS': {'name': 'iShares Euro Stoxx Mid', 'target': 0.30},
    'EXH9.DE': {'name': 'iShares Stoxx 600 Util', 'target': 0.129},
    'ISPA.DE': {'name': 'iShares Global Div 100', 'target': 0.171},
    'IUSM.DE': {'name': 'iShares Treasury 7-10yr', 'target': 0.30},
    'SYBJ.DE': {'name': 'SPDR Euro High Yield', 'target': 0.10}
}
TOLERANCE = 0.10 # Banda del 10%

# --- 3. LÓGICA DE NEGOCIO ---
def get_market_data(tickers):
    # Descarga optimizada
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3) # 3 años de historia
    
    try:
        data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
        # Manejo robusto de columnas (yfinance cambia a veces)
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close']
        elif 'Close' in data.columns:
            return data['Close']
        else:
            return data
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return pd.DataFrame()

def calculate_stats(series):
    if series.empty: return 0, 0, 0, pd.Series()
    
    # Cálculos financieros vectorizados
    ret = series.pct_change().dropna()
    cum_ret = (1 + ret).cumprod()
    
    # CAGR
    years = (series.index[-1] - series.index[0]).days / 365.25
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (1 + total_ret)**(1/years) - 1
    
    # Drawdown
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    
    # Sharpe (Risk Free 3%)
    rf = 0.03
    excess_ret = ret - (rf/252)
    sharpe = np.sqrt(252) * excess_ret.mean() / ret.std()
    
    return cagr, max_dd, sharpe, dd

# --- 4. INTERFAZ PRINCIPAL ---
st.title("📊 Dashboard de Control de Cartera")

# Sidebar limpio
with st.sidebar:
    st.header("Configuración")
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    st.info("Los datos se obtienen en tiempo real (o con 15 min de retraso) de Yahoo Finance.")

# Cargar datos
tickers = list(PORTFOLIO_CONFIG.keys())
prices_df = get_market_data(tickers)

if not prices_df.empty:
    # Limpieza de datos (Forward Fill para cubrir festivos locales diferentes)
    prices_df = prices_df.fillna(method='ffill').dropna()
    
    # --- MOTOR DE CÁLCULO ---
    # 1. Simular acciones compradas el DÍA 1 con los pesos exactos
    initial_prices = prices_df.iloc[0]
    latest_prices = prices_df.iloc[-1]
    last_date = prices_df.index[-1].strftime('%d/%m/%Y')
    
    portfolio_shares = {}
    current_market_value = 0
    holdings_data = []
    
    # Construir serie histórica de la cartera
    history_df = pd.DataFrame(index=prices_df.index)
    history_df['Total'] = 0
    
    for t in tickers:
        # Acciones teóricas (Capital * Peso / Precio Inicial)
        target_w = PORTFOLIO_CONFIG[t]['target']
        shares = (capital * target_w) / initial_prices[t]
        portfolio_shares[t] = shares
        
        # Valor actual
        val = shares * latest_prices[t]
        current_market_value += val
        
        # Sumar al histórico
        history_df['Total'] += prices_df[t] * shares

    # --- VISUALIZACIÓN DE KPIs ---
    cagr, max_dd, sharpe, dd_series = calculate_stats(history_df['Total'])
    abs_return = current_market_value - capital
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor Actual", f"{current_market_value:,.0f} €", f"{abs_return:+,.0f} €")
    col2.metric("CAGR (Anual)", f"{cagr:.2%}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}", delta_color="inverse")
    col4.metric("Ratio Sharpe", f"{sharpe:.2f}")
    
    st.caption(f"📅 Última actualización de precios: {last_date}")
    
    # --- GRÁFICOS (SUPERPUESTOS Y LIMPIOS) ---
    st.markdown("### 📈 Rendimiento y Riesgo")
    
    # Crear figura con 2 ejes (Subplots) compartiendo eje X
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3]) # 70% gráfico precio, 30% drawdown
    
    # Gráfico 1: Curva de Valor (Arriba)
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df['Total'],
        mode='lines', name='Valor Cartera',
        line=dict(color='#38bdf8', width=2), # Azul celeste claro
        fill='tozeroy', 
