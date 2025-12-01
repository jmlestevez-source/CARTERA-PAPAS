import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta

# --- 1. CONFIGURACIÓN VISUAL (ESTILO CLARO/PROFESIONAL) ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* =============================================
       1. PANEL PRINCIPAL (FONDO CLARO)
       ============================================= */
    .stApp { 
        background-color: #f8fafc !important;
    }
    
    .main h1, .main h2, .main h3 {
        color: #1e293b !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
    }
    
    .main p, .main li, .main div {
        color: #334155;
    }

    /* =============================================
       2. BARRA LATERAL (SIDEBAR) - OSCURA
       ============================================= */
    section[data-testid="stSidebar"] {
        min-width: 350px !important;
        width: 350px !important;
        background-color: #0f172a !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }

    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background-color: #1e293b !important; 
        border: 1px solid #475569 !important;
    }
    input[class*="st-"] {
        color: #ffffff !important;
    }
    div[data-baseweb="select"] svg, div[data-testid="stDateInput"] svg {
        fill: white !important;
    }
    
    section[data-testid="stSidebar"] button {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: bold;
        border: none !important;
    }

    /* =============================================
       3. TARJETAS DE MÉTRICAS (ESTILO BANCA)
       ============================================= */
    section[data-testid="stMain"] div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    section[data-testid="stMain"] div[data-testid="stMetricValue"] div {
        font-size: 2.4rem !important; 
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    section[data-testid="stMain"] div[data-testid="stMetricLabel"] p {
        font-size: 1.1rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] div {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] svg {
        transform: scale(1.3);
        margin-right: 5px;
    }

    /* =============================================
       4. TABLAS Y GRÁFICOS
       ============================================= */
    .stDataFrame { 
        border: 1px solid #cbd5e1; 
    }
    
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS DE LA CARTERA ---
# ETFs con acciones compradas y precio de compra
PORTFOLIO_CONFIG = {
    'IQQM.DE': {
        'name': 'iShares EURO STOXX Mid',
        'shares': 27,
        'buy_price': 78.25,
        'dividend_months': [6, 12],  # Semestral
        'dividend_per_share_annual': 1.50,  # Estimado anual
        'withholding': 0.00,  # Domiciliado en Irlanda
    },
    'TDIV.AS': {
        'name': 'Vanguard Dividend Leaders',
        'shares': 83,
        'buy_price': 46.72,
        'dividend_months': [3, 6, 9, 12],  # Trimestral
        'dividend_per_share_annual': 1.75,
        'withholding': 0.00,
    },
    'EHDV.DE': {
        'name': 'Invesco Euro High Div',
        'shares': 59,
        'buy_price': 31.60,
        'dividend_months': [3, 6, 9, 12],  # Trimestral
        'dividend_per_share_annual': 1.40,
        'withholding': 0.00,
    },
    'IUSM.DE': {
        'name': 'iShares Treasury 7-10yr',
        'shares': 20,
        'buy_price': 151.51,
        'dividend_months': [2, 8],  # Semestral
        'dividend_per_share_annual': 0.15,  # Bajo yield en bonos
        'withholding': 0.00,
    },
    'JNKE.MI': {
        'name': 'SPDR Euro High Yield',
        'shares': 37,
        'buy_price': 52.13,
        'dividend_months': [6, 12],  # Semestral
        'dividend_per_share_annual': 2.50,
        'withholding': 0.00,
    }
}

# Calcular capital invertido y pesos
CAPITAL_INVERTIDO = sum(cfg['shares'] * cfg['buy_price'] for cfg in PORTFOLIO_CONFIG.values())

for ticker, cfg in PORTFOLIO_CONFIG.items():
    invested = cfg['shares'] * cfg['buy_price']
    cfg['target'] = invested / CAPITAL_INVERTIDO

# Retención española sobre dividendos
RETENCION_ESPANA = 0.19

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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    
    # Fecha por defecto: 01/12/2025
    default_date = date(2025, 12, 1)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_date)
    
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
    
    st.markdown("---")
    st.subheader("📊 Cartera Actual")
    st.caption(f"Capital invertido: €{CAPITAL_INVERTIDO:,.2f}")
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        inv = cfg['shares'] * cfg['buy_price']
        st.caption(f"{ticker}: {cfg['shares']} acc × €{cfg['buy_price']:.2f} = €{inv:,.2f}")

# --- 5. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera")

# Crear tabs
tab1, tab2 = st.tabs(["📈 Rendimiento", "📅 Calendario de Dividendos"])

tickers = list(PORTFOLIO_CONFIG.keys())
with st.spinner('Actualizando precios...'):
    full_df = get_market_data_cached(tickers)

# --- TAB 1: RENDIMIENTO ---
with tab1:
    if not full_df.empty:
        full_df.index = pd.to_datetime(full_df.index)
        
        df_analysis = full_df[full_df.index >= pd.to_datetime(start_date)].copy()
        df_analysis = df_analysis.ffill().dropna()

        if len(df_analysis) == 0:
            last_known = full_df.ffill().iloc[-1]
            df_analysis = pd.DataFrame([last_known], index=[pd.to_datetime(start_date)])

        # --- MOTOR ---
        initial_prices = df_analysis.ffill().iloc[0]
        latest_prices = df_analysis.ffill().iloc[-1]
        
        portfolio_series = pd.DataFrame(index=df_analysis.index)
        portfolio_series['Total'] = 0
        portfolio_shares = {}
        invested_cash = 0
        
        for t in tickers:
            n_shares = PORTFOLIO_CONFIG[t]['shares']
            buy_price = PORTFOLIO_CONFIG[t]['buy_price']
                
            portfolio_shares[t] = n_shares
            invested_cash += n_shares * buy_price
            portfolio_series['Total'] += df_analysis[t] * n_shares
            
        cash_leftover = capital - invested_cash
        portfolio_series['Total'] += cash_leftover
        current_total = portfolio_series['Total'].iloc[-1]
        
        # --- KPIs ---
        cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(portfolio_series['Total'], capital)
        abs_ret = current_total - capital
        pct_ret = (current_total / capital) - 1

        # --- VISUALIZACIÓN ---
        k1, k2, k3, k4 = st.columns(4)
        
        k1.metric("Valor Actual", f"{current_total:,.0f} €", f"Inv: {capital:,.0f} €", delta_color="off")
        k2.metric(f"Rentabilidad (CAGR: {cagr_real:.1%})", f"{pct_ret:+.2%}", f"{abs_ret:+,.0f} €")
        k3.metric("Drawdown", f"{dd_series.iloc[-1]:.2%}", f"Max: {max_dd_real:.2%}", delta_color="inverse")
        k4.metric("Ratio Sharpe", f"{sharpe_real:.2f}")
        
        st.markdown("---")
        
        col_graph, col_table = st.columns([2, 1])
        
        with col_graph:
            st.subheader("📈 Evolución")
            
            if len(df_analysis) > 0:
                start_plot_date = portfolio_series.index[0] - timedelta(days=1)
                
                row_init_val = pd.DataFrame({'Total': [capital]}, index=[start_plot_date])
                plot_series_val = pd.concat([row_init_val, portfolio_series[['Total']]]).sort_index()
                
                row_init_dd = pd.Series([0.0], index=[start_plot_date])
                plot_series_dd = pd.concat([row_init_dd, dd_series]).sort_index()
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                fig.add_trace(go.Scatter(
                    x=plot_series_val.index, y=plot_series_val['Total'], 
                    name="Valor", mode='lines',
                    line=dict(color='#2563eb', width=3),
                    fill=None 
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=plot_series_dd.index, y=plot_series_dd, 
                    name="DD", mode='lines',
                    line=dict(color='#dc2626', width=1), 
                    fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.1)'
                ), row=2, col=1)
                
                fig.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                  showlegend=False, hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0), 
                                  font=dict(color='#334155'))
                
                fig.update_yaxes(gridcolor='#e2e8f0', row=1, col=1, autorange=True)
                fig.update_yaxes(tickformat=".0%", gridcolor='#e2e8f0', row=2, col=1)
                fig.update_xaxes(gridcolor='#e2e8f0')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Cargando...")

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
                elif w_real < min_w:
                    status = "🔵 COMPRAR"
                
                rebal_data.append({
                    "Ticker": t, "Acc.": n_shares, "Valor": f"{val_act:,.0f}€",
                    "Peso": f"{w_real:.1%}", "Estado": status
                })
                
            df_rb = pd.DataFrame(rebal_data)
            
            def style_rebal(v):
                if "VENDER" in v: return 'color: #991b1b; background-color: #fee2e2; font-weight: bold; border-radius: 4px; padding: 2px;'
                if "COMPRAR" in v: return 'color: #1e40af; background-color: #dbeafe; font-weight: bold; border-radius: 4px; padding: 2px;'
                return 'color: #166534; background-color: #dcfce7; font-weight: bold; border-radius: 4px; padding: 2px;'
                
            st.dataframe(df_rb.style.applymap(style_rebal, subset=['Estado']), use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            <div style="background-color:#ffffff; padding:15px; border-radius:8px; margin-top:20px; border:1px solid #cbd5e1; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <span style="color:#475569; font-size: 1.1rem; font-weight: 600;">Liquidez (Cash):</span>
                <span style="color:#0f172a; font-weight:bold; float:right; font-size:1.3rem;">{cash_leftover:.2f} €</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ Error de conexión.")

# --- TAB 2: CALENDARIO DE DIVIDENDOS ---
with tab2:
    st.subheader("📅 Calendario Anual de Dividendos")
    
    # Información fiscal
    st.markdown("""
    <div style="background-color:#dbeafe; border-left:4px solid #2563eb; padding:15px; border-radius:0 8px 8px 0; margin-bottom:20px;">
        <b>ℹ️ Información para inversores españoles:</b><br>
        Los ETFs UCITS domiciliados en Irlanda/Luxemburgo no tienen retención en origen. 
        En España se aplica una <b>retención del 19%</b> sobre los dividendos.
    </div>
    """, unsafe_allow_html=True)
    
    # Calcular dividendos por mes
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    dividend_calendar = []
    monthly_totals = {i: 0 for i in range(1, 13)}
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        annual_div = cfg['dividend_per_share_annual'] * cfg['shares']
        payments_per_year = len(cfg['dividend_months'])
        div_per_payment = annual_div / payments_per_year
        
        row = {
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Acciones': cfg['shares'],
            'Div/Acc Anual': f"€{cfg['dividend_per_share_annual']:.2f}",
        }
        
        for i, mes in enumerate(meses, 1):
            if i in cfg['dividend_months']:
                row[mes] = f"€{div_per_payment:.2f}"
                monthly_totals[i] += div_per_payment
            else:
                row[mes] = "-"
        
        row['Total Anual'] = f"€{annual_div:.2f}"
        dividend_calendar.append(row)
    
    # Mostrar resumen
    total_bruto = sum(cfg['dividend_per_share_annual'] * cfg['shares'] for cfg in PORTFOLIO_CONFIG.values())
    total_retencion = total_bruto * RETENCION_ESPANA
    total_neto = total_bruto - total_retencion
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Dividendo Anual Bruto", f"€{total_bruto:,.2f}")
    col2.metric("🏦 Retención España (19%)", f"-€{total_retencion:,.2f}")
    col3.metric("💰 Dividendo Anual Neto", f"€{total_neto:,.2f}")
    col4.metric("📊 Yield sobre coste", f"{(total_bruto/CAPITAL_INVERTIDO)*100:.2f}%")
    
    st.markdown("---")
    
    # Calendario visual mensual
    st.markdown("### 📆 Distribución Mensual")
    
    cols = st.columns(6)
    for i in range(12):
        mes_num = i + 1
        with cols[i % 6]:
            total_mes = monthly_totals[mes_num]
            neto_mes = total_mes * (1 - RETENCION_ESPANA)
            
            if total_mes > 0:
                bg_color = "#dcfce7"
                border_color = "#16a34a"
                icon = "💰"
            else:
                bg_color = "#f1f5f9"
                border_color = "#94a3b8"
                icon = "📅"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; border:2px solid {border_color}; 
                        border-radius:10px; padding:12px; margin:5px 0; text-align:center; min-height:120px;">
                <div style="font-size:1.3rem;">{icon}</div>
                <div style="font-weight:bold; color:#1e293b; font-size:1rem;">{meses[i]}</div>
                <div style="color:#16a34a; font-weight:600; font-size:1.1rem;">€{total_mes:.2f}</div>
                <div style="color:#64748b; font-size:0.8rem;">Neto: €{neto_mes:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabla detallada
    st.markdown("### 📋 Detalle por ETF")
    
    df_div = pd.DataFrame(dividend_calendar)
    
    def highlight_dividend(val):
        if val != "-" and "€" in str(val) and float(val.replace("€", "")) > 0:
            return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
        return ''
    
    # Aplicar estilo solo a columnas de meses
    styled_df = df_div.style.applymap(highlight_dividend, subset=meses)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Resumen fiscal
    st.markdown("---")
    st.markdown("### 🏛️ Resumen Fiscal")
    
    fiscal_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        annual_div = cfg['dividend_per_share_annual'] * cfg['shares']
        ret_origen = annual_div * cfg['withholding']
        ret_esp = annual_div * RETENCION_ESPANA
        neto = annual_div - ret_origen - ret_esp
        
        fiscal_data.append({
            'ETF': ticker,
            'Dividendo Bruto': f"€{annual_div:.2f}",
            'Ret. Origen': f"€{ret_origen:.2f} ({cfg['withholding']*100:.0f}%)",
            'Ret. España': f"€{ret_esp:.2f} (19%)",
            'Dividendo Neto': f"€{neto:.2f}",
        })
    
    df_fiscal = pd.DataFrame(fiscal_data)
    st.dataframe(df_fiscal, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background-color:#fef3c7; border-left:4px solid #f59e0b; padding:15px; border-radius:0 8px 8px 0; margin-top:20px;">
        <b>⚠️ Nota:</b> Los importes de dividendos son <b>estimaciones</b> basadas en yields históricos. 
        Los dividendos reales pueden variar. Los ETFs domiciliados en Irlanda/Luxemburgo generalmente 
        no aplican retención en origen para inversores de la UE.
    </div>
    """, unsafe_allow_html=True)
