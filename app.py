import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import calendar

# --- 1. CONFIGURACIÓN VISUAL (ESTILO CLARO/PROFESIONAL) ---
st.set_page_config(page_title="Cartera Dividendos Pro", layout="wide", page_icon="💰")

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
        min-width: 380px !important;
        width: 380px !important;
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
       3. TARJETAS DE MÉTRICAS
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
        font-size: 2.2rem !important; 
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    section[data-testid="stMain"] div[data-testid="stMetricLabel"] p {
        font-size: 1rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] div {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] svg {
        transform: scale(1.2);
        margin-right: 5px;
    }

    /* =============================================
       4. TABLAS Y CALENDARIO DIVIDENDOS
       ============================================= */
    .stDataFrame { 
        border: 1px solid #cbd5e1; 
    }
    
    .dividend-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    
    .dividend-month {
        background-color: #ffffff;
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    
    .dividend-month.active {
        background-color: #d1fae5;
        border-color: #059669;
    }
    
    .tax-info {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS DE LA CARTERA ---
PORTFOLIO_CONFIG = {
    'IQQM.DE': {  # iShares EURO STOXX Mid Cap
        'name': 'iShares EURO STOXX Mid',
        'shares': 27,
        'buy_price': 78.25,
        'dividend_yield': 0.025,  # ~2.5% anual estimado
        'dividend_months': [1, 7],  # Enero y Julio (semestral)
        'dividend_per_share': 1.95,  # Estimado anual por acción
        'country': 'DE',
        'withholding_origin': 0.26375,  # Retención Alemania
    },
    'TDIV.AS': {  # Vanguard Dividend Leaders
        'name': 'Vanguard Dividend Leaders',
        'shares': 83,
        'buy_price': 46.72,
        'dividend_yield': 0.038,  # ~3.8% anual
        'dividend_months': [3, 6, 9, 12],  # Trimestral
        'dividend_per_share': 1.78,  # Estimado anual por acción
        'country': 'NL',
        'withholding_origin': 0.15,  # Retención Países Bajos
    },
    'EHDV.DE': {  # Invesco Euro High Dividend
        'name': 'Invesco Euro High Div',
        'shares': 59,
        'buy_price': 31.60,
        'dividend_yield': 0.045,  # ~4.5% anual
        'dividend_months': [3, 6, 9, 12],  # Trimestral
        'dividend_per_share': 1.42,  # Estimado anual por acción
        'country': 'DE',
        'withholding_origin': 0.26375,  # Retención Alemania
    },
    'IUSM.DE': {  # iShares USD Treasury 7-10yr
        'name': 'iShares Treasury 7-10yr',
        'shares': 20,
        'buy_price': 151.51,
        'dividend_yield': 0.035,  # ~3.5% (cupones bonos)
        'dividend_months': [2, 8],  # Semestral
        'dividend_per_share': 5.30,  # Estimado anual por acción
        'country': 'DE',
        'withholding_origin': 0.26375,  # Retención Alemania
    },
    'JNKE.MI': {  # SPDR Euro High Yield
        'name': 'SPDR Euro High Yield',
        'shares': 37,
        'buy_price': 52.13,
        'dividend_yield': 0.048,  # ~4.8% anual
        'dividend_months': [6, 12],  # Semestral
        'dividend_per_share': 2.50,  # Estimado anual por acción
        'country': 'IT',
        'withholding_origin': 0.26,  # Retención Italia
    }
}

# Calcular capital inicial y pesos
CAPITAL_INICIAL = sum(cfg['shares'] * cfg['buy_price'] for cfg in PORTFOLIO_CONFIG.values())

for ticker, cfg in PORTFOLIO_CONFIG.items():
    invested = cfg['shares'] * cfg['buy_price']
    cfg['target'] = invested / CAPITAL_INICIAL

# Retención española
RETENCION_ESPANA = 0.19

BENCHMARK_STATS = {
    "CAGR": "5.2%", "Sharpe": "0.48", "Volatilidad": "9.5%", "Max DD": "-18.5%"
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

def get_dividend_calendar():
    """Genera el calendario de dividendos con información fiscal"""
    dividend_data = []
    monthly_totals = {i: {'bruto': 0, 'neto_origen': 0, 'neto_final': 0} for i in range(1, 13)}
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        annual_dividend = cfg['dividend_per_share'] * cfg['shares']
        payments_per_year = len(cfg['dividend_months'])
        dividend_per_payment = annual_dividend / payments_per_year
        
        for month in cfg['dividend_months']:
            # Retención en origen
            retencion_origen = dividend_per_payment * cfg['withholding_origin']
            neto_origen = dividend_per_payment - retencion_origen
            
            # Retención española (sobre el bruto, pero con crédito por doble imposición)
            # Simplificación: el neto final considera ambas retenciones
            retencion_esp = dividend_per_payment * RETENCION_ESPANA
            
            # El inversor puede deducir la menor de: retención origen o retención española
            credito_doble_imp = min(retencion_origen, retencion_esp)
            neto_final = dividend_per_payment - retencion_origen - retencion_esp + credito_doble_imp
            
            dividend_data.append({
                'Mes': month,
                'Mes_Nombre': calendar.month_abbr[month],
                'Ticker': ticker,
                'ETF': cfg['name'],
                'Div. Bruto': dividend_per_payment,
                'Ret. Origen': retencion_origen,
                'País': cfg['country'],
                '% Ret. Origen': cfg['withholding_origin'],
                'Neto Origen': neto_origen,
                'Ret. España': retencion_esp,
                'Neto Final': neto_final
            })
            
            monthly_totals[month]['bruto'] += dividend_per_payment
            monthly_totals[month]['neto_origen'] += neto_origen
            monthly_totals[month]['neto_final'] += neto_final
    
    return pd.DataFrame(dividend_data), monthly_totals

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.markdown(f"**Capital Invertido:** €{CAPITAL_INICIAL:,.2f}")
    
    # Fecha por defecto: 01/12/2025
    default_date = date(2025, 12, 1)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_date)
    
    if st.button("🔄 Recargar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Composición Cartera")
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        invested = cfg['shares'] * cfg['buy_price']
        st.markdown(f"""
        <div style="background:#1e293b; padding:8px; border-radius:6px; margin:5px 0; border-left:3px solid #10b981;">
            <span style="color:#94a3b8; font-size:0.85rem;">{ticker}</span><br>
            <span style="color:#f8fafc; font-weight:600;">{cfg['name']}</span><br>
            <span style="color:#10b981;">{cfg['shares']} acc. × €{cfg['buy_price']:.2f} = €{invested:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📜 Benchmark Histórico")
    
    col_b1, col_b2 = st.columns(2)
    col_b1.metric("CAGR", BENCHMARK_STATS["CAGR"])
    col_b1.metric("Max DD", BENCHMARK_STATS["Max DD"])
    col_b2.metric("Sharpe", BENCHMARK_STATS["Sharpe"])
    col_b2.metric("Volat.", BENCHMARK_STATS["Volatilidad"])

# --- 5. LÓGICA PRINCIPAL ---
st.title("💰 Dashboard de Cartera de Dividendos")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📈 Rendimiento", "📅 Calendario Dividendos", "📋 Detalle Fiscal"])

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
            
            if t in df_analysis.columns:
                portfolio_series['Total'] += df_analysis[t] * n_shares
            else:
                portfolio_series['Total'] += buy_price * n_shares
            
        current_total = portfolio_series['Total'].iloc[-1]
        
        # --- KPIs ---
        cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(
            portfolio_series['Total'], CAPITAL_INICIAL
        )
        abs_ret = current_total - CAPITAL_INICIAL
        pct_ret = (current_total / CAPITAL_INICIAL) - 1

        # Dividendos anuales estimados
        div_df, monthly_totals = get_dividend_calendar()
        total_div_anual_bruto = div_df['Div. Bruto'].sum()
        total_div_anual_neto = div_df['Neto Final'].sum()
        yield_on_cost = total_div_anual_bruto / CAPITAL_INICIAL

        # --- VISUALIZACIÓN KPIs ---
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("Valor Actual", f"{current_total:,.0f} €", f"Inv: {CAPITAL_INICIAL:,.0f} €", delta_color="off")
        k2.metric(f"Rentabilidad", f"{pct_ret:+.2%}", f"{abs_ret:+,.0f} €")
        k3.metric("Div. Anual Bruto", f"{total_div_anual_bruto:,.0f} €", f"Yield: {yield_on_cost:.2%}")
        k4.metric("Div. Anual Neto", f"{total_div_anual_neto:,.0f} €", f"Tras retenciones")
        k5.metric("Ratio Sharpe", f"{sharpe_real:.2f}", f"DD: {max_dd_real:.2%}")
        
        st.markdown("---")
        
        col_graph, col_table = st.columns([2, 1])
        
        with col_graph:
            st.subheader("📈 Evolución del Valor")
            
            if len(df_analysis) > 0:
                start_plot_date = portfolio_series.index[0] - timedelta(days=1)
                
                row_init_val = pd.DataFrame({'Total': [CAPITAL_INICIAL]}, index=[start_plot_date])
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
            st.subheader("⚖️ Estado Posiciones")
            
            rebal_data = []
            BAND_ABS = 0.10
            
            for t in tickers:
                target = PORTFOLIO_CONFIG[t]['target']
                n_shares = portfolio_shares[t]
                buy_price = PORTFOLIO_CONFIG[t]['buy_price']
                
                if t in latest_prices.index:
                    p_now = float(latest_prices[t]) if not pd.isna(latest_prices[t]) else buy_price
                else:
                    p_now = buy_price
                    
                val_act = n_shares * p_now
                val_compra = n_shares * buy_price
                pnl = val_act - val_compra
                pnl_pct = (val_act / val_compra - 1) if val_compra > 0 else 0
                
                w_real = val_act / current_total if current_total > 0 else 0
                
                min_w = max(0, target - BAND_ABS)
                max_w = target + BAND_ABS
                
                if w_real > max_w:
                    status = "🔴 VENDER"
                elif w_real < min_w:
                    status = "🔵 COMPRAR"
                else:
                    status = "✅ OK"
                
                rebal_data.append({
                    "Ticker": t,
                    "Acc.": n_shares,
                    "P.Compra": f"€{buy_price:.2f}",
                    "P.Actual": f"€{p_now:.2f}",
                    "Valor": f"€{val_act:,.0f}",
                    "P&L": f"{pnl:+,.0f}€ ({pnl_pct:+.1%})",
                    "Peso": f"{w_real:.1%}",
                    "Estado": status
                })
                
            df_rb = pd.DataFrame(rebal_data)
            
            def style_rebal(v):
                if "VENDER" in str(v): return 'color: #991b1b; background-color: #fee2e2; font-weight: bold;'
                if "COMPRAR" in str(v): return 'color: #1e40af; background-color: #dbeafe; font-weight: bold;'
                if "OK" in str(v): return 'color: #166534; background-color: #dcfce7; font-weight: bold;'
                return ''
                
            st.dataframe(df_rb.style.applymap(style_rebal, subset=['Estado']), use_container_width=True, hide_index=True)

    else:
        st.error("⚠️ Error de conexión con Yahoo Finance.")

# --- TAB 2: CALENDARIO DIVIDENDOS ---
with tab2:
    st.subheader("📅 Calendario Anual de Dividendos")
    
    div_df, monthly_totals = get_dividend_calendar()
    
    # Resumen anual
    total_bruto = div_df['Div. Bruto'].sum()
    total_neto = div_df['Neto Final'].sum()
    total_retencion = total_bruto - total_neto
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Anual Bruto", f"€{total_bruto:,.2f}")
    col2.metric("🏦 Total Retenciones", f"€{total_retencion:,.2f}", f"-{(total_retencion/total_bruto)*100:.1f}%")
    col3.metric("💰 Total Anual Neto", f"€{total_neto:,.2f}")
    col4.metric("📊 Yield Neto", f"{(total_neto/CAPITAL_INICIAL)*100:.2f}%")
    
    st.markdown("---")
    
    # Calendario visual
    st.markdown("### 📆 Distribución Mensual")
    
    months_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Crear grid de meses
    cols = st.columns(6)
    for i, month in enumerate(range(1, 13)):
        col_idx = i % 6
        with cols[col_idx]:
            bruto = monthly_totals[month]['bruto']
            neto = monthly_totals[month]['neto_final']
            
            if bruto > 0:
                color_bg = "#d1fae5"
                color_border = "#059669"
                icon = "💰"
            else:
                color_bg = "#f1f5f9"
                color_border = "#cbd5e1"
                icon = "📅"
            
            st.markdown(f"""
            <div style="background-color:{color_bg}; border:2px solid {color_border}; 
                        border-radius:10px; padding:15px; margin:5px 0; text-align:center;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-weight:bold; color:#1e293b; font-size:1.1rem;">{months_es[month-1]}</div>
                <div style="color:#059669; font-weight:600; font-size:1.2rem;">€{bruto:.2f}</div>
                <div style="color:#64748b; font-size:0.85rem;">Neto: €{neto:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detalle por ETF y mes
    st.markdown("### 📋 Detalle por ETF y Mes")
    
    # Crear pivot table para visualización
    pivot_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        row = {'ETF': f"{ticker} - {cfg['name']}"}
        annual_div = cfg['dividend_per_share'] * cfg['shares']
        payments = len(cfg['dividend_months'])
        per_payment = annual_div / payments
        
        for m in range(1, 13):
            if m in cfg['dividend_months']:
                row[months_es[m-1]] = f"€{per_payment:.2f}"
            else:
                row[months_es[m-1]] = "-"
        row['Total Anual'] = f"€{annual_div:.2f}"
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    def highlight_dividends(val):
        if val != "-" and val != "" and "€" in str(val):
            return 'background-color: #d1fae5; color: #059669; font-weight: bold;'
        return ''
    
    st.dataframe(
        pivot_df.style.applymap(highlight_dividends, subset=months_es),
        use_container_width=True,
        hide_index=True
    )

# --- TAB 3: DETALLE FISCAL ---
with tab3:
    st.subheader("🏛️ Información Fiscal para Inversores Españoles")
    
    # Información general
    st.markdown("""
    <div style="background-color:#fef3c7; border-left:4px solid #f59e0b; padding:20px; border-radius:0 8px 8px 0; margin:15px 0;">
        <h4 style="color:#92400e; margin:0 0 10px 0;">⚠️ Aviso Importante</h4>
        <p style="color:#78350f; margin:0;">
        Los dividendos de ETFs extranjeros están sujetos a <b>doble imposición</b>: retención en el país de origen 
        y tributación en España. Puede solicitar la <b>devolución del exceso de retención</b> en la declaración de la renta 
        mediante la deducción por doble imposición internacional.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabla de retenciones por país
    st.markdown("### 🌍 Retenciones por País de Origen")
    
    country_info = {
        'DE': {'name': 'Alemania 🇩🇪', 'rate': 26.375, 'convenio': 15.0},
        'NL': {'name': 'Países Bajos 🇳🇱', 'rate': 15.0, 'convenio': 15.0},
        'IT': {'name': 'Italia 🇮🇹', 'rate': 26.0, 'convenio': 15.0},
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (code, info) in enumerate(country_info.items()):
        with [col1, col2, col3][i]:
            exceso = info['rate'] - info['convenio']
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:15px; height:200px;">
                <h4 style="color:#1e293b; margin:0 0 10px 0;">{info['name']}</h4>
                <p style="color:#64748b; margin:5px 0;"><b>Retención estándar:</b> {info['rate']}%</p>
                <p style="color:#64748b; margin:5px 0;"><b>Límite convenio:</b> {info['convenio']}%</p>
                <p style="color:#dc2626; margin:5px 0;"><b>Exceso recuperable:</b> {exceso:.2f}%</p>
                <p style="color:#059669; margin:10px 0 0 0; font-size:0.85rem;">
                    ✅ Puede solicitar devolución del {exceso:.2f}% al país origen
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detalle fiscal por ETF
    st.markdown("### 💶 Detalle Fiscal por ETF")
    
    fiscal_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        annual_div = cfg['dividend_per_share'] * cfg['shares']
        ret_origen = annual_div * cfg['withholding_origin']
        neto_origen = annual_div - ret_origen
        
        # Retención España sobre bruto
        ret_esp = annual_div * RETENCION_ESPANA
        
        # Crédito por doble imposición (límite: el menor entre retención origen y española)
        credito = min(ret_origen, ret_esp)
        
        # Exceso de retención en origen (recuperable con formulario del país)
        limite_convenio = annual_div * 0.15  # 15% es el límite típico de convenios
        exceso_origen = max(0, ret_origen - limite_convenio)
        
        neto_final = annual_div - ret_origen - ret_esp + credito
        
        fiscal_data.append({
            'ETF': ticker,
            'Nombre': cfg['name'],
            'País': cfg['country'],
            'Dividendo Bruto': f"€{annual_div:.2f}",
            'Ret. Origen': f"€{ret_origen:.2f} ({cfg['withholding_origin']*100:.1f}%)",
            'Ret. España': f"€{ret_esp:.2f} (19%)",
            'Crédito Doble Imp.': f"€{credito:.2f}",
            'Exceso Recuperable': f"€{exceso_origen:.2f}",
            'Neto Estimado': f"€{neto_final:.2f}"
        })
    
    fiscal_df = pd.DataFrame(fiscal_data)
    st.dataframe(fiscal_df, use_container_width=True, hide_index=True)
    
    # Resumen fiscal total
    st.markdown("---")
    st.markdown("### 📊 Resumen Fiscal Anual")
    
    total_bruto = sum(cfg['dividend_per_share'] * cfg['shares'] for cfg in PORTFOLIO_CONFIG.values())
    total_ret_origen = sum(cfg['dividend_per_share'] * cfg['shares'] * cfg['withholding_origin'] for cfg in PORTFOLIO_CONFIG.values())
    total_ret_esp = total_bruto * RETENCION_ESPANA
    total_credito = sum(min(cfg['dividend_per_share'] * cfg['shares'] * cfg['withholding_origin'], 
                           cfg['dividend_per_share'] * cfg['shares'] * RETENCION_ESPANA) for cfg in PORTFOLIO_CONFIG.values())
    total_neto = total_bruto - total_ret_origen - total_ret_esp + total_credito
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                    border-radius:12px; padding:25px; color:white;">
            <h3 style="margin:0 0 20px 0; color:white;">📥 Ingresos</h3>
            <div style="display:flex; justify-content:space-between; margin:10px 0;">
                <span>Dividendos Brutos:</span>
                <span style="font-weight:bold; font-size:1.2rem;">€{total_bruto:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #991b1b 0%, #dc2626 100%); 
                    border-radius:12px; padding:25px; color:white;">
            <h3 style="margin:0 0 20px 0; color:white;">📤 Retenciones</h3>
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span>Retención Origen:</span>
                <span style="font-weight:bold;">-€{total_ret_origen:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span>Retención España:</span>
                <span style="font-weight:bold;">-€{total_ret_esp:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:8px 0; color:#86efac;">
                <span>Crédito Doble Imposición:</span>
                <span style="font-weight:bold;">+€{total_credito:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #059669 0%, #10b981 100%); 
                border-radius:12px; padding:25px; color:white; margin-top:20px; text-align:center;">
        <h3 style="margin:0 0 10px 0; color:white;">💰 DIVIDENDO NETO ANUAL ESTIMADO</h3>
        <div style="font-size:2.5rem; font-weight:bold;">€{total_neto:,.2f}</div>
        <div style="font-size:1.1rem; opacity:0.9;">Rendimiento neto sobre coste: {(total_neto/CAPITAL_INICIAL)*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Notas finales
    st.markdown("""
    <div style="background:#f1f5f9; border-radius:8px; padding:20px; margin-top:30px;">
        <h4 style="color:#475569;">📝 Notas Importantes:</h4>
        <ul style="color:#64748b;">
            <li>Los importes de dividendos son <b>estimaciones</b> basadas en yields históricos.</li>
            <li>La retención en España del 19% se aplica en la declaración de la renta.</li>
            <li>Para recuperar el exceso de retención en origen, debe presentar el formulario correspondiente al país (ej: formulario para Alemania).</li>
            <li>El crédito por doble imposición se aplica automáticamente en la declaración.</li>
            <li>Consulte con un asesor fiscal para su situación particular.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
