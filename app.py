import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import requests
from bs4 import BeautifulSoup

# --- 1. CONFIGURACIÓN VISUAL (ESTILO CLARO/PROFESIONAL) ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
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

    .stDataFrame { 
        border: 1px solid #cbd5e1; 
    }
    
</style>
""", unsafe_allow_html=True)

# --- 2. DATOS DE LA CARTERA ---
# Mapeo de tickers yfinance a URLs de Investing.com para dividendos
INVESTING_DIVIDEND_URLS = {
    'IQQM.DE': 'https://es.investing.com/etfs/ishares-dj-euro-stoxx-midcap-dividends',
    'TDIV.AS': 'https://es.investing.com/etfs/vanguard-ftse-all-wld-hidiv-yld-usd-dis-dividends',
    'EHDV.DE': 'https://es.investing.com/etfs/powershares-euro-stoxx-high-div-low-vol-dividends',
    'IUSM.DE': 'https://es.investing.com/etfs/ishares-usd-treasury-bond-7-10-yr-dividends',
    'JNKE.MI': 'https://es.investing.com/etfs/spdr-barclays-euro-high-yield-bond-dividends',
}

PORTFOLIO_CONFIG = {
    'IQQM.DE': {
        'name': 'iShares EURO STOXX Mid',
        'shares': 27,
        'buy_price': 78.25,
        'withholding': 0.00,
    },
    'TDIV.AS': {
        'name': 'Vanguard Dividend Leaders',
        'shares': 83,
        'buy_price': 46.72,
        'withholding': 0.00,
    },
    'EHDV.DE': {
        'name': 'Invesco Euro High Div',
        'shares': 59,
        'buy_price': 31.60,
        'withholding': 0.00,
    },
    'IUSM.DE': {
        'name': 'iShares Treasury 7-10yr',
        'shares': 20,
        'buy_price': 151.51,
        'withholding': 0.00,
    },
    'JNKE.MI': {
        'name': 'SPDR Euro High Yield',
        'shares': 37,
        'buy_price': 52.13,
        'withholding': 0.00,
    }
}

# Calcular capital invertido y pesos
CAPITAL_INVERTIDO = sum(cfg['shares'] * cfg['buy_price'] for cfg in PORTFOLIO_CONFIG.values())

for ticker, cfg in PORTFOLIO_CONFIG.items():
    invested = cfg['shares'] * cfg['buy_price']
    cfg['target'] = invested / CAPITAL_INVERTIDO

RETENCION_ESPANA = 0.19

BENCHMARK_STATS = {
    "CAGR": "6.81%", 
    "Sharpe": "0.529", 
    "Volatilidad": "9.40%", 
    "Max DD": "-26.76%"
}

BENCHMARK_OPTIONS = {
    "Ninguno": None,
    "MSCI World (IWDA.AS)": "IWDA.AS",
    "S&P 500 (SPY)": "SPY",
    "Euro Stoxx 50 (SX5E.DE)": "SX5E.DE",
    "Global Aggregate Bond (AGGH.MI)": "AGGH.MI",
    "MSCI Europe (IMEU.AS)": "IMEU.AS",
    "Nasdaq 100 (QQQ)": "QQQ",
}

# --- 3. FUNCIONES ---
@st.cache_data(ttl=86400, show_spinner=False)
def scrape_investing_dividends(url):
    """
    Hace scraping de la tabla de dividendos de Investing.com
    Devuelve lista de diccionarios con fecha ex-dividendo, dividendo e importe
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Buscar la tabla de dividendos
        dividends = []
        
        # Método 1: Buscar tabla con clase específica
        table = soup.find('table', {'class': 'genTbl'}) or soup.find('table', {'id': 'dividendsHistoryData'})
        
        if table:
            rows = table.find_all('tr')[1:]  # Saltar header
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    try:
                        ex_date_str = cols[0].get_text(strip=True)
                        dividend_str = cols[1].get_text(strip=True)
                        
                        # Parsear fecha
                        try:
                            ex_date = pd.to_datetime(ex_date_str, dayfirst=True)
                        except:
                            try:
                                ex_date = pd.to_datetime(ex_date_str)
                            except:
                                continue
                        
                        # Parsear dividendo
                        dividend_str = dividend_str.replace(',', '.').replace('€', '').replace('$', '').strip()
                        dividend = float(dividend_str)
                        
                        dividends.append({
                            'ex_date': ex_date,
                            'amount': dividend,
                            'month': ex_date.month
                        })
                    except (ValueError, IndexError):
                        continue
        
        # Método 2: Buscar en data attributes o scripts
        if not dividends:
            # Buscar en scripts JSON embebidos
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'dividendHistory' in script.string:
                    # Parsear JSON si existe
                    pass
        
        return dividends
    
    except Exception as e:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def get_dividend_data_from_investing(ticker):
    """
    Obtiene datos de dividendos de Investing.com para un ticker
    """
    url = INVESTING_DIVIDEND_URLS.get(ticker)
    if not url:
        return None
    
    dividends = scrape_investing_dividends(url)
    
    if dividends:
        # Filtrar últimos 12-18 meses para calcular dividendo anual
        cutoff_date = datetime.now() - timedelta(days=548)  # ~18 meses
        recent_divs = [d for d in dividends if d['ex_date'] >= cutoff_date]
        
        if recent_divs:
            # Calcular dividendo anual sumando los últimos pagos
            # Necesitamos identificar un ciclo completo
            annual_div = sum(d['amount'] for d in recent_divs[:12])  # Máximo últimos 12 pagos
            
            # Si hay menos de 4 pagos, probablemente es semestral o anual
            if len(recent_divs) <= 2:
                # Semestral: multiplicar por factor si solo tenemos parte del año
                months_covered = (datetime.now() - recent_divs[-1]['ex_date']).days / 30
                if months_covered < 12:
                    annual_div = annual_div * (12 / max(6, months_covered))
            
            # Detectar meses de pago
            div_months = sorted(list(set(d['month'] for d in recent_divs)))
            
            return {
                'annual_dividend': annual_div,
                'dividend_months': div_months,
                'history': recent_divs,
                'last_dividend': recent_divs[0]['amount'] if recent_divs else 0,
                'last_date': recent_divs[0]['ex_date'].strftime('%Y-%m-%d') if recent_divs else 'N/A',
                'source': 'Investing.com'
            }
    
    return None

# Dividendos de respaldo (datos manuales actualizados de justETF/fichas producto)
FALLBACK_DIVIDENDS = {
    'IQQM.DE': {
        'annual_dividend': 1.82,
        'dividend_months': [3, 9],
        'last_dividend': 0.91,
        'last_date': '2024-09-18',
        'history': [
            {'ex_date': datetime(2024, 9, 18), 'amount': 0.91, 'month': 9},
            {'ex_date': datetime(2024, 3, 20), 'amount': 0.89, 'month': 3},
            {'ex_date': datetime(2023, 9, 20), 'amount': 0.85, 'month': 9},
            {'ex_date': datetime(2023, 3, 22), 'amount': 0.82, 'month': 3},
        ],
        'source': 'justETF (manual)'
    },
    'TDIV.AS': {
        'annual_dividend': 1.64,
        'dividend_months': [3, 6, 9, 12],
        'last_dividend': 0.41,
        'last_date': '2024-12-18',
        'history': [
            {'ex_date': datetime(2024, 12, 18), 'amount': 0.42, 'month': 12},
            {'ex_date': datetime(2024, 9, 18), 'amount': 0.41, 'month': 9},
            {'ex_date': datetime(2024, 6, 19), 'amount': 0.40, 'month': 6},
            {'ex_date': datetime(2024, 3, 20), 'amount': 0.41, 'month': 3},
        ],
        'source': 'justETF (manual)'
    },
    'EHDV.DE': {
        'annual_dividend': 1.48,
        'dividend_months': [3, 6, 9, 12],
        'last_dividend': 0.37,
        'last_date': '2024-12-18',
        'history': [
            {'ex_date': datetime(2024, 12, 18), 'amount': 0.38, 'month': 12},
            {'ex_date': datetime(2024, 9, 18), 'amount': 0.37, 'month': 9},
            {'ex_date': datetime(2024, 6, 19), 'amount': 0.36, 'month': 6},
            {'ex_date': datetime(2024, 3, 20), 'amount': 0.37, 'month': 3},
        ],
        'source': 'justETF (manual)'
    },
    'IUSM.DE': {
        'annual_dividend': 5.30,
        'dividend_months': [4, 10],
        'last_dividend': 2.65,
        'last_date': '2024-10-16',
        'history': [
            {'ex_date': datetime(2024, 10, 16), 'amount': 2.68, 'month': 10},
            {'ex_date': datetime(2024, 4, 17), 'amount': 2.62, 'month': 4},
            {'ex_date': datetime(2023, 10, 18), 'amount': 2.55, 'month': 10},
            {'ex_date': datetime(2023, 4, 19), 'amount': 2.48, 'month': 4},
        ],
        'source': 'justETF (manual)'
    },
    'JNKE.MI': {
        'annual_dividend': 2.76,
        'dividend_months': [6, 12],
        'last_dividend': 1.38,
        'last_date': '2024-12-18',
        'history': [
            {'ex_date': datetime(2024, 12, 18), 'amount': 1.40, 'month': 12},
            {'ex_date': datetime(2024, 6, 19), 'amount': 1.36, 'month': 6},
            {'ex_date': datetime(2023, 12, 20), 'amount': 1.35, 'month': 12},
            {'ex_date': datetime(2023, 6, 21), 'amount': 1.32, 'month': 6},
        ],
        'source': 'justETF (manual)'
    }
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_dividend_data():
    """
    Obtiene datos de dividendos para todos los ETFs
    Primero intenta Investing.com, si falla usa datos de respaldo
    """
    dividend_data = {}
    
    for ticker in PORTFOLIO_CONFIG.keys():
        # Intentar obtener de Investing.com
        inv_data = get_dividend_data_from_investing(ticker)
        
        if inv_data and inv_data.get('annual_dividend', 0) > 0:
            dividend_data[ticker] = inv_data
        else:
            # Usar datos de respaldo
            dividend_data[ticker] = FALLBACK_DIVIDENDS.get(ticker, {
                'annual_dividend': 0,
                'dividend_months': [],
                'last_dividend': 0,
                'last_date': 'N/A',
                'history': [],
                'source': 'No disponible'
            })
    
    return dividend_data

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

@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_data(ticker, start_date):
    """Obtiene datos del benchmark seleccionado"""
    try:
        actual_start = pd.to_datetime(start_date) - timedelta(days=10)
        data = yf.download(ticker, start=actual_start, progress=False, auto_adjust=True)
        
        if data.empty:
            return pd.Series(dtype=float)
        
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                series = data['Close'].iloc[:, 0]
            else:
                series = data.iloc[:, 0]
        elif 'Close' in data.columns:
            series = data['Close']
        else:
            series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            
        return series
    except Exception as e:
        return pd.Series(dtype=float)

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

# --- 4. CARGAR DATOS DE DIVIDENDOS ---
with st.spinner('Cargando datos de dividendos...'):
    DIVIDEND_DATA = get_all_dividend_data()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    
    default_date = date(2025, 12, 1)
    start_date = st.date_input("Fecha Inicio Inversión", value=default_date)
    
    st.markdown("---")
    
    st.subheader("📊 Benchmark Comparativo")
    benchmark_selection = st.selectbox(
        "Seleccionar índice:",
        options=list(BENCHMARK_OPTIONS.keys()),
        index=0
    )
    
    custom_benchmark = st.text_input(
        "O introduce un ticker personalizado:",
        placeholder="Ej: VWCE.DE, ^GSPC, TEF.MC"
    )
    
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
    st.subheader("📋 Cartera Actual")
    st.caption(f"Capital invertido: €{CAPITAL_INVERTIDO:,.2f}")
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        inv = cfg['shares'] * cfg['buy_price']
        st.caption(f"{ticker}: {cfg['shares']} acc × €{cfg['buy_price']:.2f} = €{inv:,.2f}")

# --- 6. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera")

benchmark_ticker = None
if custom_benchmark.strip():
    benchmark_ticker = custom_benchmark.strip().upper()
elif BENCHMARK_OPTIONS[benchmark_selection]:
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_selection]

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
        
        cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(portfolio_series['Total'], capital)
        abs_ret = current_total - capital
        pct_ret = (current_total / capital) - 1

        k1, k2, k3, k4 = st.columns(4)
        
        k1.metric("Valor Actual", f"{current_total:,.0f} €", f"Inv: {capital:,.0f} €", delta_color="off")
        k2.metric(f"Rentabilidad (CAGR: {cagr_real:.1%})", f"{pct_ret:+.2%}", f"{abs_ret:+,.0f} €")
        k3.metric("Drawdown", f"{dd_series.iloc[-1]:.2%}", f"Max: {max_dd_real:.2%}", delta_color="inverse")
        k4.metric("Ratio Sharpe", f"{sharpe_real:.2f}")
        
        st.markdown("---")
        
        col_graph, col_table = st.columns([2, 1])
        
        with col_graph:
            if benchmark_ticker:
                st.subheader(f"📈 Evolución: Mi Cartera vs {benchmark_ticker}")
            else:
                st.subheader("📈 Evolución")
            
            if len(df_analysis) > 0:
                start_plot_date = portfolio_series.index[0] - timedelta(days=1)
                
                row_init_val = pd.DataFrame({'Total': [capital]}, index=[start_plot_date])
                plot_series_val = pd.concat([row_init_val, portfolio_series[['Total']]]).sort_index()
                
                row_init_dd = pd.Series([0.0], index=[start_plot_date])
                plot_series_dd = pd.concat([row_init_dd, dd_series]).sort_index()
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                fig.add_trace(go.Scatter(
                    x=plot_series_val.index, 
                    y=plot_series_val['Total'], 
                    name="Mi Cartera", 
                    mode='lines',
                    line=dict(color='#2563eb', width=3),
                    hovertemplate='Mi Cartera: €%{y:,.0f}<extra></extra>'
                ), row=1, col=1)
                
                benchmark_added = False
                benchmark_current_value = None
                
                if benchmark_ticker:
                    with st.spinner(f'Cargando benchmark {benchmark_ticker}...'):
                        benchmark_data = get_benchmark_data(benchmark_ticker, start_plot_date)
                    
                    if benchmark_data is not None and len(benchmark_data) > 0:
                        try:
                            benchmark_data.index = pd.to_datetime(benchmark_data.index)
                            benchmark_data = benchmark_data.sort_index()
                            
                            start_dt = pd.to_datetime(start_date)
                            
                            benchmark_before_start = benchmark_data[benchmark_data.index <= start_dt]
                            if len(benchmark_before_start) > 0:
                                benchmark_initial_price = float(benchmark_before_start.iloc[-1])
                            else:
                                benchmark_initial_price = float(benchmark_data.iloc[0])
                            
                            if benchmark_initial_price > 0:
                                benchmark_shares = capital / benchmark_initial_price
                                
                                common_dates = plot_series_val.index.intersection(benchmark_data.index)
                                
                                if len(common_dates) > 0:
                                    benchmark_aligned = benchmark_data.loc[common_dates]
                                    benchmark_value_series = benchmark_aligned * benchmark_shares
                                    
                                    benchmark_value_series = pd.concat([
                                        pd.Series([capital], index=[start_plot_date]),
                                        benchmark_value_series
                                    ]).sort_index()
                                    
                                    benchmark_value_series = benchmark_value_series.reindex(
                                        plot_series_val.index, method='ffill'
                                    ).ffill().bfill()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=benchmark_value_series.index, 
                                        y=benchmark_value_series, 
                                        name=f"{benchmark_ticker} (€{capital:,.0f} inv.)", 
                                        mode='lines',
                                        line=dict(color='#f59e0b', width=2, dash='dot'),
                                        hovertemplate=f'{benchmark_ticker}: €%{{y:,.0f}}<extra></extra>'
                                    ), row=1, col=1)
                                    
                                    benchmark_added = True
                                    benchmark_current_value = float(benchmark_value_series.iloc[-1])
                                    
                        except Exception as e:
                            st.warning(f"⚠️ Error procesando benchmark {benchmark_ticker}: {str(e)}")
                    else:
                        st.warning(f"⚠️ No se pudieron obtener datos para {benchmark_ticker}")
                
                fig.add_trace(go.Scatter(
                    x=plot_series_dd.index, y=plot_series_dd, 
                    name="DD", mode='lines',
                    line=dict(color='#dc2626', width=1), 
                    fill='tozeroy', fillcolor='rgba(220, 38, 38, 0.1)',
                    hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
                ), row=2, col=1)
                
                fig.update_layout(
                    height=520, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=12)
                    ),
                    hovermode="x unified", 
                    margin=dict(l=0,r=0,t=30,b=0), 
                    font=dict(color='#334155')
                )
                
                fig.update_yaxes(title_text="Valor (€)", gridcolor='#e2e8f0', row=1, col=1, autorange=True, tickformat=",")
                fig.update_yaxes(title_text="DD", tickformat=".0%", gridcolor='#e2e8f0', row=2, col=1)
                fig.update_xaxes(gridcolor='#e2e8f0')
                
                st.plotly_chart(fig, use_container_width=True)
                
                if benchmark_ticker and benchmark_added and benchmark_current_value is not None:
                    portfolio_ret = (float(current_total) - capital) / capital
                    benchmark_ret = (benchmark_current_value - capital) / capital
                    diff_euros = float(current_total) - benchmark_current_value
                    outperformance = portfolio_ret - benchmark_ret
                    
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    col_p1.metric("📊 Mi Cartera", f"€{current_total:,.0f}", f"{portfolio_ret:+.2%}")
                    col_p2.metric(f"📈 {benchmark_ticker}", f"€{benchmark_current_value:,.0f}", f"{benchmark_ret:+.2%}")
                    
                    if diff_euros >= 0:
                        col_p3.metric("💰 Diferencia", f"+€{diff_euros:,.0f}")
                        col_p4.metric("🏆 Outperformance", f"+{outperformance:.2%}")
                    else:
                        col_p3.metric("💸 Diferencia", f"€{diff_euros:,.0f}")
                        col_p4.metric("📉 Underperformance", f"{outperformance:.2%}")

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
    
    st.markdown("""
    <div style="background-color:#dbeafe; border-left:4px solid #2563eb; padding:15px; border-radius:0 8px 8px 0; margin-bottom:20px;">
        <b>ℹ️ Información para inversores españoles:</b><br>
        Los ETFs UCITS domiciliados en Irlanda/Luxemburgo <b>no tienen retención en origen</b>. 
        En España se aplica una <b>retención del 19%</b> sobre los dividendos cobrados.
    </div>
    """, unsafe_allow_html=True)
    
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Calcular totales y preparar datos
    dividend_calendar = []
    monthly_totals = {i: 0 for i in range(1, 13)}
    total_bruto = 0
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        div_months = div_data.get('dividend_months', [])
        annual_div = div_per_share * cfg['shares']
        total_bruto += annual_div
        
        payments_per_year = len(div_months) if div_months else 1
        div_per_payment = annual_div / payments_per_year if payments_per_year > 0 else 0
        
        yield_on_cost = (div_per_share / cfg['buy_price']) * 100 if cfg['buy_price'] > 0 else 0
        
        row = {
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Acciones': cfg['shares'],
            'Div/Acc': f"€{div_per_share:.2f}",
            'Yield': f"{yield_on_cost:.1f}%",
        }
        
        for i, mes in enumerate(meses, 1):
            if i in div_months:
                row[mes] = f"€{div_per_payment:.2f}"
                monthly_totals[i] += div_per_payment
            else:
                row[mes] = "-"
        
        row['Total Anual'] = f"€{annual_div:.2f}"
        row['Fuente'] = div_data.get('source', 'N/A')
        dividend_calendar.append(row)
    
    total_retencion = total_bruto * RETENCION_ESPANA
    total_neto = total_bruto - total_retencion
    yield_cartera = (total_bruto / CAPITAL_INVERTIDO) * 100 if CAPITAL_INVERTIDO > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Dividendo Anual Bruto", f"€{total_bruto:,.2f}")
    col2.metric("🏦 Retención España (19%)", f"-€{total_retencion:,.2f}")
    col3.metric("💰 Dividendo Anual Neto", f"€{total_neto:,.2f}")
    col4.metric("📊 Yield sobre coste", f"{yield_cartera:.2f}%")
    
    st.markdown("---")
    
    # Tabla de dividendos con historial
    st.markdown("### 📊 Dividendos por ETF (Datos de Investing.com)")
    
    div_detail = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        annual_div = div_per_share * cfg['shares']
        yield_on_cost = (div_per_share / cfg['buy_price']) * 100 if cfg['buy_price'] > 0 else 0
        last_div = div_data.get('last_dividend', 0)
        last_date = div_data.get('last_date', 'N/A')
        
        div_detail.append({
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Acciones': cfg['shares'],
            'P. Compra': f"€{cfg['buy_price']:.2f}",
            'Div/Acc Anual': f"€{div_per_share:.2f}",
            'Último Div.': f"€{last_div:.2f}",
            'Fecha Último': last_date,
            'Total Anual': f"€{annual_div:.2f}",
            'Yield': f"{yield_on_cost:.2f}%",
            'Fuente': div_data.get('source', 'N/A')
        })
    
    st.dataframe(pd.DataFrame(div_detail), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Historial de dividendos detallado
    st.markdown("### 📜 Historial de Pagos Recientes")
    
    history_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA.get(ticker, {})
        history = div_data.get('history', [])
        
        for payment in history[:4]:  # Últimos 4 pagos
            if isinstance(payment, dict):
                ex_date = payment.get('ex_date', '')
                if isinstance(ex_date, datetime):
                    ex_date = ex_date.strftime('%Y-%m-%d')
                
                amount = payment.get('amount', 0)
                total_payment = amount * cfg['shares']
                
                history_data.append({
                    'ETF': ticker,
                    'Fecha Ex-Div': ex_date,
                    'Div/Acc': f"€{amount:.4f}",
                    'Acciones': cfg['shares'],
                    'Total Bruto': f"€{total_payment:.2f}",
                    'Total Neto': f"€{total_payment * (1-RETENCION_ESPANA):.2f}"
                })
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        df_history = df_history.sort_values('Fecha Ex-Div', ascending=False)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    
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
    
    # Tabla calendario detallada
    st.markdown("### 📋 Calendario Detallado por ETF")
    
    df_div = pd.DataFrame(dividend_calendar)
    
    def highlight_dividend(val):
        if val != "-" and "€" in str(val):
            try:
                if float(val.replace("€", "")) > 0:
                    return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
            except:
                pass
        return ''
    
    styled_df = df_div.style.applymap(highlight_dividend, subset=meses)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Resumen fiscal
    st.markdown("---")
    st.markdown("### 🏛️ Resumen Fiscal Anual")
    
    fiscal_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        annual_div = div_per_share * cfg['shares']
        ret_origen = annual_div * cfg['withholding']
        ret_esp = annual_div * RETENCION_ESPANA
        neto = annual_div - ret_origen - ret_esp
        
        fiscal_data.append({
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Dividendo Bruto': f"€{annual_div:.2f}",
            'Ret. Origen': f"€{ret_origen:.2f} ({cfg['withholding']*100:.0f}%)",
            'Ret. España (19%)': f"€{ret_esp:.2f}",
            'Dividendo Neto': f"€{neto:.2f}",
        })
    
    fiscal_data.append({
        'ETF': '📊 TOTAL',
        'Nombre': '',
        'Dividendo Bruto': f"€{total_bruto:.2f}",
        'Ret. Origen': f"€0.00 (0%)",
        'Ret. España (19%)': f"€{total_retencion:.2f}",
        'Dividendo Neto': f"€{total_neto:.2f}",
    })
    
    df_fiscal = pd.DataFrame(fiscal_data)
    st.dataframe(df_fiscal, use_container_width=True, hide_index=True)
    
    monthly_net_avg = total_neto / 12
    st.markdown(f"""
    <div style="background-color:#dcfce7; border:2px solid #16a34a; padding:20px; border-radius:10px; margin-top:20px; text-align:center;">
        <span style="color:#166534; font-size:1.2rem; font-weight:600;">💰 Ingreso Mensual Medio (Neto):</span>
        <span style="color:#166534; font-weight:bold; font-size:1.8rem; margin-left:15px;">€{monthly_net_avg:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#fef3c7; border-left:4px solid #f59e0b; padding:15px; border-radius:0 8px 8px 0; margin-top:20px;">
        <b>📝 Fuentes de datos:</b><br>
        • Los dividendos se obtienen mediante scraping de <b>Investing.com</b> (historial de pagos reales)<br>
        • Si no están disponibles, se usan datos de respaldo de <b>justETF</b> y fichas de producto<br>
        • Los ETFs UCITS domiciliados en Irlanda no aplican retención en origen para inversores de la UE
    </div>
    """, unsafe_allow_html=True)
