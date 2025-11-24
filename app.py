import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta

# --- 1. CONFIGURACIÓN VISUAL Y ESTILOS ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

# CSS para mejorar legibilidad y contraste (Estilo "Slate Blue")
st.markdown("""
<style>
    /* Fondo App */
    .stApp {
        background-color: #1a202c; /* Gris muy oscuro azulado */
    }
    
    /* Textos */
    h1, h2, h3 { color: #e2e8f0 !important; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    p, div, label, span { color: #cbd5e0 !important; }
    
    /* Métricas (Tarjetas) */
    div[data-testid="stMetric"] {
        background-color: #2d3748;
        border-left: 5px solid #3182ce; /* Borde azul */
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 1.6rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171923;
    }
    
    /* Tablas */
    .stDataFrame { border: 1px solid #4a5568; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE DATOS ---
PORTFOLIO_CONFIG = {
    'DJMC.AS': {'name': 'iShares Euro Stoxx Mid', 'target': 0.30},
    'EXH9.DE': {'name': 'iShares Stoxx 600 Util', 'target': 0.129},
    'ISPA.DE': {'name': 'iShares Global Div 100', 'target': 0.171},
    'IUSM.DE': {'name': 'iShares Treasury 7-10yr', 'target': 0.30},
    'SYBJ.DE': {'name': 'SPDR Euro High Yield', 'target': 0.10}
}

# Datos de tu Backtest (Benchmark Histórico)
BENCHMARK_STATS = {
    "CAGR": "6.77%",
    "Sharpe": "0.572",
    "Volatilidad": "8.77%",
    "Max DD": "-21.19%"
}

# --- 3. FUNCIONES ROBUSTAS ---
def get_market_data(tickers, start_date):
    # Descargamos un poco antes para asegurar datos en la fecha de inicio
    download_start = start_date - timedelta(days=7)
    try:
        data = yf.download(tickers, start=download_start, progress=False, auto_adjust=True)
        
        # Manejo de MultiIndex (Problema común de yfinance reciente)
        if isinstance(data.columns, pd.MultiIndex):
            # Intentamos extraer 'Close', si no existe, usamos el nivel 0
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close']
            else:
                df = data.iloc[:, :len(tickers)] # Fallback brusco
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            df = data # Si ya viene plano
            
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def calculate_metrics(series, capital_inicial):
    if series.empty: return 0, 0, 0, pd.Series()
    
    # Retornos diarios
    ret = series.pct_change().dropna()
    
    # CAGR (Basado en fecha inicio seleccionada)
    days = (series.index[-1] - series.index[0]).days
    if days > 0:
        years = days / 365.25
        total_ret = (series.iloc[-1] / capital_inicial) - 1
        cagr = (1 + total_ret)**(1/years) - 1
    else:
        cagr = 0
    
    # Drawdown
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    
    # Sharpe (RF 3%)
    rf = 0.03
    if ret.std() > 0:
        excess_ret = ret - (rf/252)
        sharpe = np.sqrt(252) * excess_ret.mean() / ret.std()
    else:
        sharpe = 0
        
    return cagr, max_dd, sharpe, dd

# --- 4. SIDEBAR (CONFIG Y COMPARATIVA) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    # Fecha por defecto: 1 año atrás
    default_start = date.today() - timedelta(days=365)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_start)
    
    st.markdown("---")
    st.header("📜 Benchmark Histórico")
    st.caption("Referencia de tu Backtest (Bandas ±10%)")
    
    # Mostrar tus métricas estáticas para comparar
    c1, c2 = st.columns(2)
    c1.metric("CAGR Teórico", BENCHMARK_STATS["CAGR"])
    c1.metric("Max DD Teórico", BENCHMARK_STATS["Max DD"])
    c2.metric("Sharpe Teórico", BENCHMARK_STATS["Sharpe"])
    c2.metric("Volatilidad", BENCHMARK_STATS["Volatilidad"])
    
    st.info("👆 Estos datos son fijos de tu estudio previo. A la derecha verás los datos reales de tu periodo seleccionado.")

# --- 5. LÓGICA PRINCIPAL ---
st.title(f"Dashboard de Cartera")
st.markdown(f"Periodo analizado: **{start_date.strftime('%d/%m/%Y')}** hasta **Hoy**")

tickers = list(PORTFOLIO_CONFIG.keys())
full_df = get_market_data(tickers, start_date)

# Filtrar estrictamente desde la fecha seleccionada
if not full_df.empty:
    # Aseguramos que el índice es datetime
    full_df.index = pd.to_datetime(full_df.index)
    # Recortamos los datos para empezar exactamente en o después de la fecha elegida
    df_analysis = full_df[full_df.index >= pd.to_datetime(start_date)].copy()
    
    # Rellenar huecos (festivos)
    df_analysis = df_analysis.fillna(method='ffill').dropna()

    if len(df_analysis) > 0:
        # 1. Motor de la Cartera (Simulación desde cero)
        initial_prices = df_analysis.iloc[0]
        latest_prices = df_analysis.iloc[-1]
        
        # Calculamos cuántas acciones hubiéramos comprado ese día con el capital
        shares = {}
        current_value = 0
        
        # Serie temporal del valor total
        portfolio_series = pd.DataFrame(index=df_analysis.index)
        portfolio_series['Total'] = 0
        
        for t in tickers:
            target_w = PORTFOLIO_CONFIG[t]['target']
            # Acciones = (Capital * Peso) / Precio Día 1
            num_shares = (capital * target_w) / initial_prices[t]
            shares[t] = num_shares
            
            # Sumar a la serie temporal
            portfolio_series['Total'] += df_analysis[t] * num_shares
            
            # Valor actual hoy
            current_value += num_shares * latest_prices[t]

        # 2. Calcular Métricas Reales
        cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(portfolio_series['Total'], capital)
        
        # 3. KPIs Superiores
        k1, k2, k3, k4 = st.columns(4)
        
        diff_eur = current_value - capital
        diff_pct = (current_value / capital) - 1
        
        k1.metric("Valor Actual", f"{current_value:,.0f} €", f"{diff_eur:+,.0f} €")
        k2.metric("Rentabilidad Total", f"{diff_pct:+.2%}", f"CAGR: {cagr_real:.2%}")
        k3.metric("Drawdown Actual", f"{dd_series.iloc[-1]:.2%}", f"Max: {max_dd_real:.2%}", delta_color="inverse")
        k4.metric("Ratio Sharpe Real", f"{sharpe_real:.2f}")

        # 4. Gráficos (Estilo Limpio)
        st.markdown("### 📈 Evolución y Riesgo")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Curva de Valor
        fig.add_trace(go.Scatter(
            x=portfolio_series.index, y=portfolio_series['Total'],
            mode='lines', name='Valor Cartera',
            line=dict(color='#63b3ed', width=2), # Azul claro
            fill='tozeroy', fillcolor='rgba(99, 179, 237, 0.1)'
        ), row=1, col=1)
        
        # Curva Drawdown
        fig.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            mode='lines', name='Drawdown',
            line=dict(color='#fc8181', width=1.5), # Rojo claro
            fill='tozeroy', fillcolor='rgba(252, 129, 129, 0.2)'
        ), row=2, col=1)
        
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            showlegend=False,
            font=dict(color="#cbd5e0"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.update_yaxes(gridcolor='#2d3748', title="Valor (€)", row=1, col=1)
        fig.update_yaxes(gridcolor='#2d3748', title="Caída %", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(gridcolor='#2d3748')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Tabla de Rebalanceo (ABSOLUTO)
        st.markdown("### ⚖️ Control de Bandas (Absoluto ±10%)")
        st.caption("Las bandas son puntos porcentuales absolutos sobre el objetivo (Ej: 30% ±10% = Rango 20% - 40%)")
        
        rebal_data = []
        BAND_ABS = 0.10 # 10% Absoluto
        
        for t in tickers:
            target = PORTFOLIO_CONFIG[t]['target']
            val_actual = shares[t] * latest_prices[t]
            weight_actual = val_actual / current_value
            
            # LOGICA ABSOLUTA
            min_w = max(0, target - BAND_ABS) # No bajar de 0
            max_w = target + BAND_ABS
            
            status = "✅ EN RANGO"
            color_status = "green"
            op = "-"
            
            if weight_actual > max_w:
                status = "🔴 VENDER"
                color_status = "red"
                # Cuanto sobra para volver al target
                surplus = val_actual - (current_value * target) 
                op = f"Vender {surplus:,.0f} €"
            elif weight_actual < min_w:
                status = "🔵 COMPRAR"
                color_status = "blue"
                # Cuanto falta para volver al target
                deficit = (current_value * target) - val_actual
                op = f"Comprar {deficit:,.0f} €"
                
            rebal_data.append({
                "Activo": PORTFOLIO_CONFIG[t]['name'],
                "Precio": f"{latest_prices[t]:.2f} €",
                "Peso Objetivo": target,
                "Banda Mín": min_w,
                "Peso Actual": weight_actual,
                "Banda Máx": max_w,
                "Acción": status,
                "Operación Sugerida": op
            })
            
        df_rb = pd.DataFrame(rebal_data)
        
        # Estilizar Tabla
        def color_rows(val):
            if "VENDER" in val: color = "#feb2b2" # Rojo pastel
            elif "COMPRAR" in val: color = "#90cdf4" # Azul pastel
            else: color = "#9ae6b4" # Verde pastel
            return f'color: black; background-color: {color}; font-weight: bold; border-radius: 4px; padding: 4px;'

        st.dataframe(
            df_rb.style
            .format({
                "Peso Objetivo": "{:.1%}", "Peso Actual": "{:.2%}", 
                "Banda Mín": "{:.0%}", "Banda Máx": "{:.0%}"
            })
            .applymap(color_rows, subset=['Acción']),
            use_container_width=True,
            height=300
        )
        
    else:
        st.warning("⚠️ Datos descargados pero vacíos para el rango de fechas seleccionado. Intenta ampliar la fecha de inicio un par de días.")
else:
    st.error("❌ Error crítico descargando datos. Yahoo Finance puede estar bloqueando las peticiones temporalmente o los Tickers son incorrectos.")
