import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# Estructura: Ticker: {Nombre, Acciones, Precio_Compra_Original, Meses_Div_Tipicos}
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
        'name': 'Invesco Euro High Div Low Vol', 
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
        'name': 'SPDR Bloomberg Euro High Yield', 
        'shares': 37, 
        'buy_price': 52.13,
        'div_months': 'Feb, Ago'
    }
}

TICKERS = list(PORTFOLIO_DATA.keys())

# --- 3. FUNCIONES ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_market_data(tickers):
    # Descargamos historia suficiente para gráficas
    start_download = datetime.now() - timedelta(days=365*3)
    try:
        data = yf.download(tickers, start=start_download, progress=False, auto_adjust=False) # auto_adjust=False para tener Close real
        # Manejo seguro de MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close']
            else:
                df = data.iloc[:, :len(tickers)] # Fallback
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            df = data
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def get_dividend_info(tickers):
    """Obtiene info de dividendos (Yield y último pago)"""
    div_info = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            # Intentar obtener yield actual
            yield_pct = info.get('dividendYield', 0)
            if yield_pct is None: yield_pct = 0
            
            div_info[t] = {
                'yield': yield_pct,
                'currency': info.get('currency', 'EUR')
            }
        except:
            div_info[t] = {'yield': 0, 'currency': 'EUR'}
    return div_info

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # FECHA POR DEFECTO MODIFICADA A 01/12/2025
    default_date = date(2025, 12, 1)
    start_date = st.date_input("Fecha Inicio Análisis", value=default_date)
    
    if start_date > date.today():
        st.warning("⚠️ La fecha seleccionada es futura. No habrá datos históricos para graficar hasta que llegue esa fecha.")

    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.info("Cartera configurada con posiciones fijas (Stock/Bonos).")

# --- 5. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera Personal")

with st.spinner('Actualizando precios y dividendos...'):
    full_df = get_market_data(TICKERS)
    div_data = get_dividend_info(TICKERS)

if not full_df.empty:
    # Preparación de datos de precios
    full_df.index = pd.to_datetime(full_df.index)
    latest_prices = full_df.ffill().iloc[-1]
    
    # --- CÁLCULOS DE POSICIÓN ---
    total_invested = 0
    total_current_value = 0
    portfolio_rows = []

    for t in TICKERS:
        data = PORTFOLIO_DATA[t]
        shares = data['shares']
        buy_price = data['buy_price']
        curr_price = latest_prices[t]
        
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

    # Totales Globales
    global_pl = total_current_value - total_invested
    global_pl_pct = (global_pl / total_invested) if total_invested > 0 else 0

    # --- INTERFAZ DE USUARIO ---
    
    # 1. KPIs Superiores
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor Total", f"{total_current_value:,.0f} €", f"Inv: {total_invested:,.0f} €")
    k2.metric("P&L Total", f"{global_pl:+,.0f} €", f"{global_pl_pct:+.2%}")
    
    # Drawdown simple (basado en historia reciente del portfolio sintético)
    # Creamos una serie histórica del valor actual de la cartera
    hist_value = pd.Series(0, index=full_df.index)
    for t in TICKERS:
        hist_value += full_df[t] * PORTFOLIO_DATA[t]['shares']
    
    # Filtramos por fecha seleccionada (si es pasada)
    if start_date <= date.today():
        hist_view = hist_value[hist_value.index >= pd.to_datetime(start_date)]
    else:
        hist_view = hist_value.tail(100) # Si fecha futura, mostramos últimos 100 días disponibles

    if not hist_view.empty:
        rolling_max = hist_view.cummax()
        dd = (hist_view - rolling_max) / rolling_max
        max_dd = dd.min()
        k3.metric("Max Drawdown (Periodo)", f"{max_dd:.2%}", "Desde Máximo")
    else:
        k3.metric("Max Drawdown", "N/A")
        
    k4.metric("Posiciones", f"{len(TICKERS)}")

    st.markdown("---")

    # --- TABS DE DETALLE ---
    tab1, tab2, tab3 = st.tabs(["📊 Resumen y Posiciones", "📈 Gráfico Evolución", "💰 Calendario Dividendos"])

    with tab1:
        st.subheader("Detalle de Posiciones")
        df_port = pd.DataFrame(portfolio_rows)
        
        # Formateo para visualización
        df_display = df_port.copy()
        df_display['Inversión'] = df_display['Inversión'].map('{:,.0f} €'.format)
        df_display['Valor Actual'] = df_display['Valor Actual'].map('{:,.0f} €'.format)
        df_display['P&L (€)'] = df_display['P&L (€)'].map('{:+,.0f} €'.format)
        df_display['P&L (%)'] = df_display['P&L (%)'].map('{:+.2%}'.format)
        
        st.dataframe(
            df_display[["Ticker", "Nombre", "Acciones", "P. Compra", "P. Actual", "Inversión", "Valor Actual", "P&L (€)", "P&L (%)"]],
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.subheader("Evolución del Valor de la Cartera")
        if not hist_view.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_view.index, y=hist_view,
                mode='lines', name='Valor Cartera',
                line=dict(color='#2563eb', width=3),
                fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'
            ))
            
            # Línea de inversión inicial (Costo base constante)
            fig.add_trace(go.Scatter(
                x=hist_view.index, y=[total_invested]*len(hist_view),
                mode='lines', name='Capital Invertido',
                line=dict(color='#64748b', width=2, dash='dash')
            ))

            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis_title="Fecha", yaxis_title="Valor (€)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos históricos suficientes para la fecha seleccionada.")

    with tab3:
        st.subheader("Estimación de Dividendos y Fiscalidad (España)")
        st.markdown("""
        <small>Cálculo estimado basado en el *Yield* anualizado actual. 
        La retención aplicada es del <b>19%</b> (IRPF base ahorro España). 
        Los importes son aproximados y pueden variar según el tipo de cambio y anuncios de la gestora.</small>
        """, unsafe_allow_html=True)
        
        div_rows = []
        total_annual_gross = 0
        total_annual_net = 0
        
        RETENCION_ESP = 0.19

        for t in TICKERS:
            shares = PORTFOLIO_DATA[t]['shares']
            curr_price = latest_prices[t]
            yield_dec = div_data[t]['yield']
            
            # Cálculo Anual Estimado
            gross_annual = (curr_price * yield_dec) * shares
            retention = gross_annual * RETENCION_ESP
            net_annual = gross_annual - retention
            
            total_annual_gross += gross_annual
            total_annual_net += net_annual
            
            div_rows.append({
                "ETF": PORTFOLIO_DATA[t]['name'],
                "Meses Típicos de Cobro": PORTFOLIO_DATA[t]['div_months'],
                "Yield Actual": f"{yield_dec:.2%}",
                "Bruto Estimado (Año)": gross_annual,
                "Retención (19%)": retention,
                "Neto Estimado (Año)": net_annual
            })
            
        df_divs = pd.DataFrame(div_rows)
        
        # Métricas de Dividendos
        d1, d2, d3 = st.columns(3)
        d1.metric("Dividendo Anual Bruto", f"{total_annual_gross:,.2f} €")
        d2.metric("Retención Hda. (19%)", f"-{total_annual_gross * RETENCION_ESP:,.2f} €", delta_color="inverse")
        d3.metric("Dividendo Anual Neto", f"{total_annual_net:,.2f} €", delta_color="normal")
        
        # Formato tabla dividendos
        st.markdown("#### Desglose por ETF")
        
        # Aplicar formato moneda para visualizar
        df_div_show = df_divs.copy()
        df_div_show["Bruto Estimado (Año)"] = df_divs["Bruto Estimado (Año)"].map('{:,.2f} €'.format)
        df_div_show["Retención (19%)"] = df_divs["Retención (19%)"].map('{:,.2f} €'.format)
        df_div_show["Neto Estimado (Año)"] = df_divs["Neto Estimado (Año)"].map('{:,.2f} €'.format)
        
        st.dataframe(df_div_show, use_container_width=True, hide_index=True)

else:
    st.error("⚠️ No se pudieron cargar los datos de mercado. Revisa tu conexión a internet.")
