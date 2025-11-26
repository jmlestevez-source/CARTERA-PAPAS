import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta

# --- 1. CONFIGURACIÓN VISUAL Y CSS DE ALTO CONTRASTE ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* --- FONDO GENERAL --- */
    .stApp { background-color: #111827; } /* Casi negro azulado */
    
    /* --- BARRA LATERAL (SIDEBAR) --- */
    section[data-testid="stSidebar"] {
        min-width: 400px !important;
        width: 400px !important;
        background-color: #1f2937 !important; /* Gris oscuro sólido */
    }
    
    /* FORZAR COLOR BLANCO EN TODOS LOS TEXTOS DEL SIDEBAR */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div {
        color: #f3f4f6 !important; /* Blanco brillante */
    }
    
    /* --- TEXTOS GENERALES --- */
    h1, h2, h3 { color: #e2e8f0 !important; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    
    /* --- TARJETAS DE MÉTRICAS --- */
    div[data-testid="stMetric"] {
        background-color: #374151; /* Gris medio */
        border-left: 4px solid #60a5fa; /* Azul brillante */
        border-radius: 6px; 
        padding: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { color: #d1d5db !important; font-size: 0.95rem !important; }
    
    /* --- TABLAS --- */
    .stDataFrame { border: 1px solid #4b5563; }
    
    /* --- Inputs --- */
    /* Asegurar que el texto dentro de las cajas de input sea legible */
    .stDateInput input, .stNumberInput input {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS DE CARTERA ---
PORTFOLIO_CONFIG = {
    'DJMC.AS': {'name': 'iShares Euro Stoxx Mid', 'target': 0.30},
    'EXH9.DE': {'name': 'iShares Stoxx 600 Util', 'target': 0.129},
    'ISPA.DE': {'name': 'iShares Global Div 100', 'target': 0.171},
    'IUSM.DE': {'name': 'iShares Treasury 7-10yr', 'target': 0.30},
    'SYBJ.DE': {'name': 'SPDR Euro High Yield', 'target': 0.10}
}

BENCHMARK_STATS = {
    "CAGR": "6.77%", "Sharpe": "0.572", "Volatilidad": "8.77%", "Max DD": "-21.19%"
}

# --- 3. FUNCIONES ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_market_data_cached(tickers):
    start_date = datetime.now() - timedelta(days=365*5)
    try:
        data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close']
            else:
                df = data.iloc[:, :len(tickers)]
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            df = data
        return df
    except Exception:
        return pd.DataFrame()

def calculate_metrics(series, capital_inicial):
    if series.empty or len(series) < 2: 
        return 0.0, 0.0, 0.0, pd.Series(0, index=series.index)
    
    ret = series.pct_change().fillna(0)
    days = (series.index[-1] - series.index[0]).days
    current_val = series.iloc[-1]
    if days > 0:
        total_ret = (current_val / capital_inicial) - 1
        cagr = (1 + total_ret)**(365.25/days) - 1 if total_ret > -0.9 else 0
    else:
        cagr = 0.0
    
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    
    rf = 0.03
    if ret.std() > 0:
        excess_ret = ret - (rf/252)
        sharpe = np.sqrt(252) * excess_ret.mean() / ret.std()
    else:
        sharpe = 0.0
        
    return cagr, max_dd, sharpe, dd

# --- 4. BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    
    default_start = date(date.today().year - 1, 1, 1)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_start)
    
    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📜 Benchmark Histórico")
    col_b1, col_b2 = st.columns(2)
    col_b1.metric("CAGR", BENCHMARK_STATS["CAGR"])
    col_b1.metric("Max DD", BENCHMARK_STATS["Max DD"])
    col_b2.metric("Sharpe", BENCHMARK_STATS["Sharpe"])
    col_b2.metric("Volat.", BENCHMARK_STATS["Volatilidad"])

# --- 5. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera")

tickers = list(PORTFOLIO_CONFIG.keys())
with st.spinner('Consultando datos de mercado...'):
    full_df = get_market_data_cached(tickers)

if not full_df.empty:
    full_df.index = pd.to_datetime(full_df.index)
    
    # Filtro de fecha
    df_analysis = full_df[full_df.index >= pd.to_datetime(start_date)].copy()
    df_analysis = df_analysis.ffill().dropna()

    # Simulación si no hay datos
    if len(df_analysis) == 0:
        st.info("📅 Esperando apertura de mercado. Simulando datos iniciales.")
        last_known = full_df.ffill().iloc[-1]
        df_analysis = pd.DataFrame([last_known], index=[pd.to_datetime(start_date)])

    # --- MOTOR DE INVERSIÓN ---
    initial_prices = df_analysis.ffill().iloc[0]
    latest_prices = df_analysis.ffill().iloc[-1]
    
    portfolio_series = pd.DataFrame(index=df_analysis.index)
    portfolio_series['Total'] = 0
    portfolio_shares = {}
    invested_cash = 0
    
    for t in tickers:
        target_w = PORTFOLIO_CONFIG[t]['target']
        budget = capital * target_w
        price_val = float(initial_prices[t])
        
        if pd.isna(price_val) or price_val <= 0:
            n_shares = 0
        else:
            n_shares = int(budget // price_val)
            
        portfolio_shares[t] = n_shares
        invested_cash += n_shares * price_val
        portfolio_series['Total'] += df_analysis[t] * n_shares
        
    cash_leftover = capital - invested_cash
    portfolio_series['Total'] += cash_leftover
    current_total = portfolio_series['Total'].iloc[-1]
    
    # --- KPIs ---
    cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(portfolio_series['Total'], capital)
    abs_ret = current_total - capital
    pct_ret = (current_total / capital) - 1

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor Actual", f"{current_total:,.0f} €", f"Inv: {capital:,.0f} €", delta_color="off")
    k2.metric(f"Rentabilidad (CAGR: {cagr_real:.1%})", f"{pct_ret:+.2%}", f"{abs_ret:+,.0f} €")
    k3.metric("Drawdown", f"{dd_series.iloc[-1]:.2%}", f"Max: {max_dd_real:.2%}", delta_color="inverse")
    k4.metric("Ratio Sharpe", f"{sharpe_real:.2f}")
    
    st.markdown("---")
    
    # --- GRÁFICOS ---
    col_graph, col_table = st.columns([2, 1])
    
    with col_graph:
        st.subheader("📈 Evolución")
        
        if len(df_analysis) > 0:
            # 1. Preparamos los datos para que empiecen visualmente en el Capital Inicial
            start_plot_date = portfolio_series.index[0] - timedelta(days=1)
            
            # Serie VALOR
            row_init_val = pd.DataFrame({'Total': [capital]}, index=[start_plot_date])
            plot_series_val = pd.concat([row_init_val, portfolio_series[['Total']]]).sort_index()
            
            # Serie DRAWDOWN
            row_init_dd = pd.Series([0.0], index=[start_plot_date])
            plot_series_dd = pd.concat([row_init_dd, dd_series]).sort_index()
            
            # 2. Configuración Plotly
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            # GRÁFICO ARRIBA (VALOR): SIN RELLENO (fill=None) para que el eje Y se ajuste automáticamente
            fig.add_trace(go.Scatter(
                x=plot_series_val.index, y=plot_series_val['Total'], 
                name="Valor", mode='lines',
                line=dict(color='#38bdf8', width=3), # Azul cian brillante
                fill=None # <--- CLAVE: Sin relleno para que el eje Y no empiece en 0
            ), row=1, col=1)
            
            # GRÁFICO ABAJO (RIESGO): CON RELLENO (fill='tozeroy')
            fig.add_trace(go.Scatter(
                x=plot_series_dd.index, y=plot_series_dd, 
                name="DD", mode='lines',
                line=dict(color='#f87171', width=1), 
                fill='tozeroy', fillcolor='rgba(248, 113, 113, 0.2)'
            ), row=2, col=1)
            
            fig.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              showlegend=False, hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0), 
                              font=dict(color='#cbd5e0'))
            
            # Ajuste Ejes
            fig.update_yaxes(gridcolor='#374151', row=1, col=1, autorange=True) # Auto-escala
            fig.update_yaxes(tickformat=".0%", gridcolor='#374151', row=2, col=1)
            fig.update_xaxes(gridcolor='#374151')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Esperando datos...")

    with col_table:
        st.subheader("⚖️ Bandas (Abs ±10%)")
        
        rebal_data = []
        BAND_ABS = 0.10
        
        for t in tickers:
            target = PORTFOLIO_CONFIG[t]['target']
            n_shares = portfolio_shares[t]
            p_now = float(latest_prices[t]) if not pd.isna(latest_prices[t]) else 0.0
            val_act = n_shares * p_now
            
            w_real = val_act / current_total if current_total > 0 else 0
            
            min_w = max(0, target - BAND_ABS)
            max_w = target + BAND_ABS
            
            status = "✅ MANTENER"
            
            if w_real > max_w:
                status = "🔴 VENDER"
                surplus = val_act - (current_total * target)
                op_txt = f"Venta: {surplus:.0f}€"
            elif w_real < min_w:
                status = "🔵 COMPRAR"
                deficit = (current_total * target) - val_act
                op_txt = f"Compra: {deficit:.0f}€"
            else:
                op_txt = "-"
            
            rebal_data.append({
                "Ticker": t, "Acc.": n_shares, "Valor": f"{val_act:,.0f}€",
                "Peso": f"{w_real:.1%}", "Estado": status
            })
            
        df_rb = pd.DataFrame(rebal_data)
        def style_rebal(v):
            if "VENDER" in v: return 'color: #fca5a5; font-weight: bold;'
            if "COMPRAR" in v: return 'color: #93c5fd; font-weight: bold;'
            return 'color: #6ee7b7;'
            
        st.dataframe(df_rb.style.applymap(style_rebal, subset=['Estado']), use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        <div style="background-color:#374151; padding:12px; border-radius:6px; margin-top:15px; border:1px solid #4b5563;">
            <span style="color:#9ca3af;">Liquidez (Cash):</span>
            <span style="color:#fff; font-weight:bold; float:right; font-size:1.1rem;">{cash_leftover:.2f} €</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("⚠️ Error de conexión con Yahoo Finance.")
