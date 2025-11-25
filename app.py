import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta

# --- 1. CONFIGURACIÓN VISUAL Y CSS AVANZADO ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* Fondo General */
    .stApp { background-color: #1a202c; }
    
    /* Forzar anchura mínima de la Sidebar */
    section[data-testid="stSidebar"] {
        min-width: 400px !important;
        width: 400px !important;
    }
    
    /* Tipografía y Colores */
    h1, h2, h3 { color: #e2e8f0 !important; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    p, div, label, span, li { color: #cbd5e0 !important; }
    
    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {
        background-color: #2d3748;
        border-left: 4px solid #3182ce;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.5rem !important; }
    div[data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.9rem !important; }
    
    /* Tablas más legibles */
    .stDataFrame { border: 1px solid #4a5568; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS MAESTROS ---
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
    
    # CAGR (Desde fecha inicio)
    days = (series.index[-1] - series.index[0]).days
    current_val = series.iloc[-1]
    
    if days > 0:
        total_ret = (current_val / capital_inicial) - 1
        cagr = (1 + total_ret)**(365.25/days) - 1 if total_ret > -0.9 else 0
    else:
        cagr = 0.0
    
    # Drawdown
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    
    # Sharpe (Anualizado)
    rf = 0.03
    if ret.std() > 0:
        excess_ret = ret - (rf/252)
        sharpe = np.sqrt(252) * excess_ret.mean() / ret.std()
    else:
        sharpe = 0.0
        
    return cagr, max_dd, sharpe, dd

# --- 4. SIDEBAR (MÁS ANCHO AHORA) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    
    # Fecha por defecto
    default_start = date(date.today().year - 1, 1, 1)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_start)
    
    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📜 Benchmark Histórico (Teórico)")
    
    # Usamos columnas dentro del sidebar ancho para organizar mejor
    col_b1, col_b2 = st.columns(2)
    col_b1.metric("CAGR", BENCHMARK_STATS["CAGR"])
    col_b1.metric("Max DD", BENCHMARK_STATS["Max DD"])
    col_b2.metric("Sharpe", BENCHMARK_STATS["Sharpe"])
    col_b2.metric("Volat.", BENCHMARK_STATS["Volatilidad"])
    
    st.caption("Nota: Estos datos son de tu estudio previo (Backtest).")

# --- 5. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera")

tickers = list(PORTFOLIO_CONFIG.keys())
with st.spinner('Analizando mercado...'):
    full_df = get_market_data_cached(tickers)

if not full_df.empty:
    full_df.index = pd.to_datetime(full_df.index)
    # Filtrar desde fecha seleccionada
    df_analysis = full_df[full_df.index >= pd.to_datetime(start_date)].copy()
    df_analysis = df_analysis.ffill().dropna()

    # Verificación: Si la fecha es HOY o FUTURA, o no hay datos aún
    if len(df_analysis) == 0:
        st.info("📅 La fecha seleccionada es hoy o no hay datos de mercado aún. Mostrando estado inicial.")
        # Simulamos un dataframe con precios de cierre de ayer para poder mostrar la tabla
        last_available = full_df.iloc[-1]
        df_analysis = pd.DataFrame([last_available], index=[pd.to_datetime(start_date)])

    # --- MOTOR DE COMPRA DE ACCIONES (ENTERAS) ---
    initial_prices = df_analysis.iloc[0]
    latest_prices = df_analysis.iloc[-1]
    
    holdings = []
    invested_cash = 0
    
    # Serie temporal
    portfolio_series = pd.DataFrame(index=df_analysis.index)
    portfolio_series['Total'] = 0
    
    # Cálculo de Acciones Enteras
    portfolio_shares = {}
    
    for t in tickers:
        target_w = PORTFOLIO_CONFIG[t]['target']
        budget_for_asset = capital * target_w
        price_at_start = initial_prices[t]
        
        # MAGIA: División entera (Floor) para obtener acciones completas
        num_shares = int(budget_for_asset // price_at_start)
        portfolio_shares[t] = num_shares
        
        cost_basis = num_shares * price_at_start
        invested_cash += cost_basis
        
        # Construir histórico
        portfolio_series['Total'] += df_analysis[t] * num_shares
        
    # Cash sobrante (Liquidez)
    cash_leftover = capital - invested_cash
    # Sumamos el cash al valor total de la cartera (asumimos que se queda en cuenta)
    portfolio_series['Total'] += cash_leftover
    
    current_total_value = portfolio_series['Total'].iloc[-1]
    
    # --- MÉTRICAS ---
    cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(portfolio_series['Total'], capital)
    
    abs_return = current_total_value - capital
    pct_return = (current_total_value / capital) - 1

    # --- VISUALIZACIÓN ---
    
    # Fila de KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor Actual", f"{current_total_value:,.0f} €", f"{abs_return:+,.0f} €")
    k2.metric("Rentabilidad", f"{pct_return:+.2%}", f"CAGR: {cagr_real:.2%}")
    k3.metric("Drawdown", f"{dd_series.iloc[-1]:.2%}", f"Max: {max_dd_real:.2%}", delta_color="inverse")
    k4.metric("Ratio Sharpe", f"{sharpe_real:.2f}")
    
    st.markdown("---")

    # Gráficos
    col_graph, col_table = st.columns([2, 1])
    
    with col_graph:
        st.subheader("📈 Evolución")
        if len(df_analysis) > 1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=portfolio_series.index, y=portfolio_series['Total'], name="Valor", 
                                     line=dict(color='#63b3ed', width=2), fill='tozeroy', fillcolor='rgba(99,179,237,0.1)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name="DD", 
                                     line=dict(color='#fc8181', width=1), fill='tozeroy', fillcolor='rgba(252,129,129,0.2)'), row=2, col=1)
            fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              showlegend=False, hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0), font=dict(color='#cbd5e0'))
            fig.update_yaxes(gridcolor='#2d3748', row=1, col=1)
            fig.update_yaxes(tickformat=".0%", gridcolor='#2d3748', row=2, col=1)
            fig.update_xaxes(gridcolor='#2d3748')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("El gráfico aparecerá mañana, cuando haya datos de variación.")

    with col_table:
        st.subheader("⚖️ Control de Bandas (Abs ±10%)")
        
        rebal_data = []
        BAND_ABS = 0.10
        
        for t in tickers:
            target = PORTFOLIO_CONFIG[t]['target']
            n_shares = portfolio_shares[t]
            val_act = n_shares * latest_prices[t]
            
            # Peso Real sobre el TOTAL (Incluyendo Cash)
            w_real = val_act / current_total_value
            
            min_w = max(0, target - BAND_ABS)
            max_w = target + BAND_ABS
            
            status = "✅ MANTENER"
            
            if w_real > max_w:
                status = "🔴 VENDER"
                surplus = val_act - (current_total_value * target)
                op_txt = f"Vender {surplus:.0f}€"
            elif w_real < min_w:
                status = "🔵 COMPRAR"
                deficit = (current_total_value * target) - val_act
                op_txt = f"Comprar {deficit:.0f}€"
            else:
                op_txt = "-"
                
            rebal_data.append({
                "Ticker": t,
                "Acciones": n_shares,
                "Precio": f"{latest_prices[t]:.2f}€",
                "Valor": f"{val_act:,.0f}€",
                "Peso": f"{w_real:.1%}",
                "Estado": status
            })
            
        df_rb = pd.DataFrame(rebal_data)
        
        def style_rebal(v):
            if "VENDER" in v: return 'color: #feb2b2; font-weight: bold;'
            if "COMPRAR" in v: return 'color: #90cdf4; font-weight: bold;'
            return 'color: #68d391;'
            
        st.dataframe(
            df_rb.style.applymap(style_rebal, subset=['Estado']),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Acciones": st.column_config.NumberColumn(format="%d"),
            }
        )
        
        st.markdown(f"""
        <div style="background-color:#2d3748; padding:10px; border-radius:5px; margin-top:10px;">
            <span style="color:#a0aec0; font-size:0.9rem;">Liquidez (Cash sobrante):</span>
            <span style="color:#fff; font-weight:bold; float:right;">{cash_leftover:.2f} €</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("⚠️ Error de conexión con Yahoo Finance. Espera unos minutos.")
