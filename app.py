import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Cartera de Inversión", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* PANEL PRINCIPAL */
    .stApp { background-color: #f8fafc !important; }
    .main h1, .main h2, .main h3 { color: #1e293b !important; font-family: 'Segoe UI', sans-serif; font-weight: 800; }
    .main p, .main li, .main div { color: #334155; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #0f172a !important; min-width: 300px !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #f8fafc !important; }
    div[data-baseweb="input"], div[data-baseweb="base-input"] { background-color: #1e293b !important; border: 1px solid #475569 !important; }
    input[class*="st-"] { color: #ffffff !important; }
    div[data-testid="stDateInput"] svg { fill: white !important; }
    section[data-testid="stSidebar"] button { background-color: #2563eb !important; color: white !important; border: none !important; }

    /* MÉTRICAS */
    div[data-testid="stMetric"] { background-color: #ffffff !important; border: 1px solid #e2e8f0; border-left: 6px solid #2563eb; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] div { font-size: 2.0rem !important; color: #0f172a !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] p { font-size: 1.0rem !important; color: #64748b !important; font-weight: 600 !important; }
    
    /* TABLAS */
    .stDataFrame { border: 1px solid #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS DE LA CARTERA ---
PORTFOLIO_DATA = {
    'iQQM.DE': {
        'name': 'iShares EURO STOXX Mid', 
        'shares': 27, 
        'buy_price': 78.25,
        'div_months': 'Mar, Jun, Sep, Dic'
    },
    'TDIV.AS': {
        'name': 'Vanguard Div. Leaders', 
        'shares': 83, 
        'buy_price': 46.72,
        'div_months': 'Mar, Jun, Sep, Dic'
    },
    'EHDV.DE': {
        'name': 'Invesco Euro High Div', 
        'shares': 59, 
        'buy_price': 31.60,
        'div_months': 'Mar, Jun, Sep, Dic'
    },
    'IUSM.DE': {
        'name': 'iShares USD Treasury 7-10yr', 
        'shares': 220, 
        'buy_price': 151.51,
        'div_months': 'Semestral (Feb, Ago) o Trim.'
    },
    'JNKE.MI': {
        'name': 'SPDR Bloomberg Euro HY', 
        'shares': 37, 
        'buy_price': 52.13,
        'div_months': 'Feb, Ago'
    }
}

TICKERS = list(PORTFOLIO_DATA.keys())

# --- 3. FUNCIONES ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_market_data(tickers):
    start_download = datetime.now() - timedelta(days=365*3)
    try:
        # Descarga masiva
        data = yf.download(tickers, start=start_download, progress=False, group_by='ticker', auto_adjust=False)
        
        # Reconstruir DataFrame solo con 'Close' para simplificar el manejo
        close_data = pd.DataFrame()
        
        for t in tickers:
            try:
                # Manejo de MultiIndex (yfinance v0.2+)
                if t in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else t in data.columns:
                    if isinstance(data.columns, pd.MultiIndex):
                        series = data[t]['Close']
                    else:
                        # Caso borde: si solo se descarga 1 ticker y no es multiindex
                        if 'Close' in data.columns:
                            series = data['Close']
                        else:
                            series = pd.Series()
                    
                    if not series.empty:
                        close_data[t] = series
            except Exception:
                continue
                
        return close_data
    except Exception as e:
        st.error(f"Error general descargando datos: {e}")
        return pd.DataFrame()

def get_dividend_info(tickers):
    div_info = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            yield_pct = info.get('dividendYield', 0)
            if yield_pct is None: yield_pct = 0
            div_info[t] = {'yield': yield_pct}
        except:
            div_info[t] = {'yield': 0}
    return div_info

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    default_date = date(2025, 12, 1)
    start_date = st.date_input("Fecha Inicio Análisis", value=default_date)
    
    if start_date > date.today():
        st.warning("⚠️ Fecha futura seleccionada. Gráficos históricos pueden aparecer vacíos.")

    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.info("Datos proporcionados por Yahoo Finance (Retraso 15min).")

# --- 5. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera Personal")

with st.spinner('Conectando con mercados...'):
    full_df = get_market_data(TICKERS)
    div_data = get_dividend_info(TICKERS)

# Comprobación de que tenemos datos
if not full_df.empty:
    full_df.index = pd.to_datetime(full_df.index)
    # Rellenar huecos de fines de semana/festivos para no tener NaNs
    full_df = full_df.ffill()
    
    # Obtener últimos precios disponibles de la serie
    try:
        latest_prices = full_df.iloc[-1]
    except:
        latest_prices = pd.Series()

    # --- CÁLCULOS DE POSICIÓN (CON PROTECCIÓN ERROR) ---
    total_invested = 0
    total_current_value = 0
    portfolio_rows = []
    
    missing_tickers = []

    for t in TICKERS:
        data = PORTFOLIO_DATA[t]
        shares = data['shares']
        buy_price = data['buy_price']
        
        # --- SOLUCIÓN DEL KEYERROR ---
        # Verificamos si el ticker existe en los precios descargados
        if t in latest_prices.index and not pd.isna(latest_prices[t]):
            curr_price = float(latest_prices[t])
        else:
            # Si falla la descarga, usamos el precio de compra para no romper la app
            curr_price = buy_price 
            missing_tickers.append(t)
        
        invested = shares * buy_price
        current_val = shares * curr_price
        pl_euro = current_val - invested
        pl_pct = (pl_euro / invested) if invested > 0 else 0
        
        total_invested += invested
        total_current_value += current_val
        
        portfolio_rows.append({
            "Ticker": t,
            "Nombre": data['name'],
            "Acciones": shares,
            "P. Compra": f"{buy_price:.2f} €",
            "P. Actual": f"{curr_price:.2f} €",
            "Inversión": invested,
            "Valor Actual": current_val,
            "P&L (€)": pl_euro,
            "P&L (%)": pl_pct
        })
        
    # Aviso si faltan datos
    if missing_tickers:
        st.warning(f"⚠️ No se pudieron descargar datos actuales para: {', '.join(missing_tickers)}. Se muestra precio de compra para evitar errores.")

    # Totales Globales
    global_pl = total_current_value - total_invested
    global_pl_pct = (global_pl / total_invested) if total_invested > 0 else 0

    # --- INTERFAZ ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor Total", f"{total_current_value:,.0f} €", f"Inv: {total_invested:,.0f} €")
    k2.metric("P&L Total", f"{global_pl:+,.0f} €", f"{global_pl_pct:+.2%}")
    
    # Cálculo Drawdown Sintético
    hist_value = pd.Series(0, index=full_df.index)
    for t in TICKERS:
        # Solo sumar al histórico si tenemos datos para ese ticker
        if t in full_df.columns:
            hist_value += full_df[t] * PORTFOLIO_DATA[t]['shares']
    
    # Filtrar fecha
    if start_date <= date.today():
        hist_view = hist_value[hist_value.index >= pd.to_datetime(start_date)]
    else:
        hist_view = hist_value.tail(100)

    if not hist_view.empty and hist_view.max() > 0:
        rolling_max = hist_view.cummax()
        dd = (hist_view - rolling_max) / rolling_max
        max_dd = dd.min()
        k3.metric("Max Drawdown", f"{max_dd:.2%}")
    else:
        k3.metric("Max Drawdown", "N/A")
        
    k4.metric("Activos", f"{len(TICKERS)}")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Posiciones", "📈 Gráfico", "💰 Dividendos"])

    with tab1:
        df_port = pd.DataFrame(portfolio_rows)
        # Formato visual
        df_display = df_port.copy()
        df_display['Inversión'] = df_display['Inversión'].map('{:,.0f} €'.format)
        df_display['Valor Actual'] = df_display['Valor Actual'].map('{:,.0f} €'.format)
        df_display['P&L (€)'] = df_display['P&L (€)'].map('{:+,.0f} €'.format)
        df_display['P&L (%)'] = df_display['P&L (%)'].map('{:+.2%}'.format)
        
        st.dataframe(
            df_display[["Ticker", "Nombre", "Acciones", "P. Compra", "P. Actual", "Valor Actual", "P&L (%)"]],
            use_container_width=True, hide_index=True
        )

    with tab2:
        if not hist_view.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_view.index, y=hist_view,
                mode='lines', name='Cartera',
                line=dict(color='#2563eb', width=3),
                fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=hist_view.index, y=[total_invested]*len(hist_view),
                mode='lines', name='Capital',
                line=dict(color='#64748b', width=2, dash='dash')
            ))
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos históricos para graficar.")

    with tab3:
        st.caption("Estimación anual basada en Yield actual. Retención IRPF España: 19%.")
        div_rows = []
        total_net = 0
        RETENCION = 0.19
        
        for t in TICKERS:
            data = PORTFOLIO_DATA[t]
            # Usamos precio actual (o compra si falló descarga)
            price_ref = buy_price
            if t in latest_prices.index and not pd.isna(latest_prices[t]):
                price_ref = float(latest_prices[t])
                
            y_pct = div_data.get(t, {}).get('yield', 0)
            
            gross = (price_ref * y_pct) * data['shares']
            net = gross * (1 - RETENCION)
            total_net += net
            
            div_rows.append({
                "ETF": data['name'],
                "Meses Pago": data['div_months'],
                "Yield": f"{y_pct:.2%}",
                "Bruto Anual": f"{gross:.2f} €",
                "Neto (Tras 19%)": f"{net:.2f} €"
            })
            
        col_d1, col_d2 = st.columns([1, 3])
        col_d1.metric("Neto Anual Estimado", f"{total_net:,.2f} €")
        with col_d2:
            st.dataframe(pd.DataFrame(div_rows), use_container_width=True, hide_index=True)

else:
    st.error("⚠️ Error crítico: No se pudieron obtener datos de Yahoo Finance. Intenta recargar en unos minutos.")
