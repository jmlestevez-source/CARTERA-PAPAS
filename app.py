import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import requests
from bs4 import BeautifulSoup
import re

# --- 1. CONFIGURACIÓN VISUAL ---
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
        min-width: 320px !important;
        width: 320px !important;
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

    /* Campos de entrada mejorados */
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background-color: #1e293b !important; 
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    input[class*="st-"] {
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    
    /* Select boxes */
    div[data-baseweb="select"] {
        background-color: #1e293b !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    div[data-baseweb="select"] svg, div[data-testid="stDateInput"] svg {
        fill: white !important;
    }
    
    /* Date input */
    div[data-testid="stDateInput"] > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    /* Number input */
    div[data-testid="stNumberInput"] > div > div > input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    section[data-testid="stSidebar"] button {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: bold;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background-color: #1d4ed8 !important;
    }

    /* Métricas en el main */
    section[data-testid="stMain"] div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    section[data-testid="stMain"] div[data-testid="stMetricValue"] div {
        font-size: 2rem !important; 
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    section[data-testid="stMain"] div[data-testid="stMetricLabel"] p {
        font-size: 1rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] div {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    .stDataFrame { 
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE LA CARTERA ---
# Fecha de compra de las acciones
FECHA_COMPRA = date(2025, 12, 1)

PORTFOLIO_CONFIG = {
    'IQQM.DE': {
        'name': 'iShares EURO STOXX Mid',
        'isin': 'IE00B02KXL92',
        'shares': 27,
        'buy_price': 78.09,
        'withholding': 0.00,
    },
    'TDIV.AS': {
        'name': 'VanEck Dividend Leaders',
        'isin': 'NL0011683594',
        'shares': 83,
        'buy_price': 46.75,
        'withholding': 0.00,
    },
    'EHDV.DE': {
        'name': 'Invesco Euro High Div Low Vol',
        'isin': 'IE00BZ4BMM98',
        'shares': 59,
        'buy_price': 31.54,
        'withholding': 0.00,
    },
    'IUSM.DE': {
        'name': 'iShares $ Treasury 7-10yr',
        'isin': 'IE00B1FZS798',
        'shares': 20,
        'buy_price': 151.20,
        'withholding': 0.00,
    },
    'JNKE.MI': {
        'name': 'SPDR Euro High Yield Bond',
        'isin': 'IE00B6YX5M31',
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
    "Telefónica (TEF.MC)": "TEF.MC",
    "Ninguno": None,
    "MSCI World (IWDA.AS)": "IWDA.AS",
    "S&P 500 (SPY)": "SPY",
    "Euro Stoxx 50 (SX5E.DE)": "SX5E.DE",
    "Global Aggregate Bond (AGGH.MI)": "AGGH.MI",
    "MSCI Europe (IMEU.AS)": "IMEU.AS",
    "Nasdaq 100 (QQQ)": "QQQ",
    "IBEX 35 (^IBEX)": "^IBEX",
}

# --- 3. FUNCIONES DE SCRAPING JUSTETF ---
@st.cache_data(ttl=86400, show_spinner=False)
def scrape_justetf_dividends(isin):
    url = f"https://www.justetf.com/es/etf-profile.html?isin={isin}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
    }
    
    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        dividend_data = {
            'annual_dividend': 0,
            'dividend_yield': 0,
            'distribution_frequency': '',
            'last_dividend': 0,
            'last_ex_date': '',
            'dividend_months': [],
            'history': [],
            'source': 'justETF'
        }
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    
                    if 'rendimiento' in label and 'dividendo' in label:
                        try:
                            yield_val = re.search(r'([\d,\.]+)\s*%', value)
                            if yield_val:
                                dividend_data['dividend_yield'] = float(yield_val.group(1).replace(',', '.'))
                        except:
                            pass
                    
                    if 'distribución' in label or 'frecuencia' in label:
                        dividend_data['distribution_frequency'] = value
        
        distribution_section = soup.find('div', {'id': 'distributions'}) or soup.find('section', {'id': 'distributions'})
        
        if distribution_section:
            dist_table = distribution_section.find('table')
            if dist_table:
                rows = dist_table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        try:
                            date_text = cols[0].get_text(strip=True)
                            amount_text = cols[1].get_text(strip=True)
                            
                            ex_date = None
                            for fmt in ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                                try:
                                    ex_date = datetime.strptime(date_text, fmt)
                                    break
                                except:
                                    continue
                            
                            amount_match = re.search(r'([\d,\.]+)', amount_text.replace(',', '.'))
                            if amount_match and ex_date:
                                amount = float(amount_match.group(1).replace(',', '.'))
                                dividend_data['history'].append({
                                    'ex_date': ex_date,
                                    'amount': amount,
                                    'month': ex_date.month
                                })
                        except:
                            continue
        
        if dividend_data['history']:
            dividend_data['history'].sort(key=lambda x: x['ex_date'], reverse=True)
            
            one_year_ago = datetime.now() - timedelta(days=400)
            recent = [d for d in dividend_data['history'] if d['ex_date'] >= one_year_ago]
            
            if recent:
                dividend_data['annual_dividend'] = sum(d['amount'] for d in recent)
                dividend_data['dividend_months'] = sorted(list(set(d['month'] for d in recent)))
                dividend_data['last_dividend'] = recent[0]['amount']
                dividend_data['last_ex_date'] = recent[0]['ex_date'].strftime('%Y-%m-%d')
        
        return dividend_data
        
    except Exception as e:
        return {
            'annual_dividend': 0,
            'dividend_yield': 0,
            'distribution_frequency': '',
            'last_dividend': 0,
            'last_ex_date': '',
            'dividend_months': [],
            'history': [],
            'source': f'Error: {str(e)}'
        }

# DATOS DE DIVIDENDOS ACTUALIZADOS (Enero 2025)
FALLBACK_DIVIDENDS = {
    'IQQM.DE': {
        'annual_dividend': 1.95,
        'dividend_yield': 2.48,
        'distribution_frequency': 'Semestral',
        'dividend_months': [3, 9],
        'last_dividend': 1.02,
        'last_ex_date': '2024-09-19',
        'history': [
            {'ex_date': datetime(2024, 9, 19), 'amount': 1.02, 'month': 9},
            {'ex_date': datetime(2024, 3, 14), 'amount': 0.93, 'month': 3},
            {'ex_date': datetime(2023, 9, 14), 'amount': 0.98, 'month': 9},
            {'ex_date': datetime(2023, 3, 16), 'amount': 0.87, 'month': 3},
        ],
        'source': 'justETF/iShares (Ene 2025)'
    },
    'TDIV.AS': {
        'annual_dividend': 1.72,
        'dividend_yield': 3.68,
        'distribution_frequency': 'Trimestral',
        'dividend_months': [3, 6, 9, 12],
        'last_dividend': 0.44,
        'last_ex_date': '2024-12-16',
        'history': [
            {'ex_date': datetime(2024, 12, 16), 'amount': 0.44, 'month': 12},
            {'ex_date': datetime(2024, 9, 16), 'amount': 0.43, 'month': 9},
            {'ex_date': datetime(2024, 6, 17), 'amount': 0.42, 'month': 6},
            {'ex_date': datetime(2024, 3, 18), 'amount': 0.43, 'month': 3},
        ],
        'source': 'justETF/VanEck (Ene 2025)'
    },
    'EHDV.DE': {
        'annual_dividend': 1.52,
        'dividend_yield': 4.82,
        'distribution_frequency': 'Trimestral',
        'dividend_months': [1, 4, 7, 10],
        'last_dividend': 0.39,
        'last_ex_date': '2024-10-17',
        'history': [
            {'ex_date': datetime(2024, 10, 17), 'amount': 0.39, 'month': 10},
            {'ex_date': datetime(2024, 7, 18), 'amount': 0.38, 'month': 7},
            {'ex_date': datetime(2024, 4, 18), 'amount': 0.37, 'month': 4},
            {'ex_date': datetime(2024, 1, 18), 'amount': 0.38, 'month': 1},
        ],
        'source': 'justETF/Invesco (Ene 2025)'
    },
    'IUSM.DE': {
        'annual_dividend': 5.45,
        'dividend_yield': 3.60,
        'distribution_frequency': 'Semestral',
        'dividend_months': [4, 10],
        'last_dividend': 2.78,
        'last_ex_date': '2024-10-10',
        'history': [
            {'ex_date': datetime(2024, 10, 10), 'amount': 2.78, 'month': 10},
            {'ex_date': datetime(2024, 4, 11), 'amount': 2.67, 'month': 4},
            {'ex_date': datetime(2023, 10, 12), 'amount': 2.58, 'month': 10},
            {'ex_date': datetime(2023, 4, 13), 'amount': 2.45, 'month': 4},
        ],
        'source': 'justETF/iShares (Ene 2025)'
    },
    'JNKE.MI': {
        'annual_dividend': 2.85,
        'dividend_yield': 5.47,
        'distribution_frequency': 'Semestral',
        'dividend_months': [6, 12],
        'last_dividend': 1.45,
        'last_ex_date': '2024-12-12',
        'history': [
            {'ex_date': datetime(2024, 12, 12), 'amount': 1.45, 'month': 12},
            {'ex_date': datetime(2024, 6, 13), 'amount': 1.40, 'month': 6},
            {'ex_date': datetime(2023, 12, 14), 'amount': 1.38, 'month': 12},
            {'ex_date': datetime(2023, 6, 15), 'amount': 1.35, 'month': 6},
        ],
        'source': 'justETF/SPDR (Ene 2025)'
    }
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_dividend_data():
    dividend_data = {}
    scraping_status = {}
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        isin = cfg.get('isin', '')
        scraped_data = scrape_justetf_dividends(isin)
        
        if scraped_data.get('annual_dividend', 0) > 0 or scraped_data.get('history'):
            dividend_data[ticker] = scraped_data
            scraping_status[ticker] = 'justETF (scraping OK)'
        else:
            fallback = FALLBACK_DIVIDENDS.get(ticker, {})
            dividend_data[ticker] = fallback
            scraping_status[ticker] = fallback.get('source', 'Datos no disponibles')
    
    return dividend_data, scraping_status

def get_dividends_from_date(dividend_data, start_date):
    """
    Filtra los dividendos para mostrar solo los que ocurrirán después de la fecha de inicio.
    Proyecta dividendos futuros basándose en los meses de pago históricos.
    """
    filtered_data = {}
    start_dt = datetime.combine(start_date, datetime.min.time())
    
    for ticker, data in dividend_data.items():
        filtered = data.copy()
        
        # Calcular meses restantes en el primer año desde la fecha de compra
        start_month = start_date.month
        start_year = start_date.year
        
        div_months = data.get('dividend_months', [])
        
        # Meses de dividendo que quedan después de la fecha de compra
        remaining_months_first_year = [m for m in div_months if m > start_month]
        
        # Calcular dividendo proporcional para el primer año
        if div_months:
            payments_per_year = len(div_months)
            remaining_payments_first_year = len(remaining_months_first_year)
            
            annual_div = data.get('annual_dividend', 0)
            div_per_payment = annual_div / payments_per_year if payments_per_year > 0 else 0
            
            # Dividendo del primer año parcial
            first_year_dividend = div_per_payment * remaining_payments_first_year
            
            filtered['first_year_dividend'] = first_year_dividend
            filtered['remaining_months_first_year'] = remaining_months_first_year
            filtered['first_year_payments'] = remaining_payments_first_year
        else:
            filtered['first_year_dividend'] = 0
            filtered['remaining_months_first_year'] = []
            filtered['first_year_payments'] = 0
        
        filtered_data[ticker] = filtered
    
    return filtered_data

# --- 4. FUNCIONES DE MERCADO ---
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

# --- 5. CARGAR DATOS ---
with st.spinner('Cargando datos de dividendos...'):
    DIVIDEND_DATA, SCRAPING_STATUS = get_all_dividend_data()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    
    # Fecha de inicio por defecto: 1 de diciembre de 2025
    start_date = st.date_input(
        "Fecha Inicio Inversión", 
        value=FECHA_COMPRA,
        help="Fecha de compra de las acciones"
    )
    
    st.markdown("---")
    
    st.subheader("📊 Benchmark")
    benchmark_selection = st.selectbox(
        "Seleccionar índice:",
        options=list(BENCHMARK_OPTIONS.keys()),
        index=0
    )
    
    custom_benchmark = st.text_input(
        "Ticker personalizado:",
        placeholder="Ej: VWCE.DE, ^GSPC"
    )
    
    if st.button("🔄 Recargar Datos", use_container_width=True):
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
    st.subheader("📋 Cartera")
    st.caption(f"📅 Fecha compra: {FECHA_COMPRA.strftime('%d/%m/%Y')}")
    st.caption(f"💰 Capital: €{CAPITAL_INVERTIDO:,.2f}")
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        inv = cfg['shares'] * cfg['buy_price']
        st.caption(f"• {ticker}: {cfg['shares']} × €{cfg['buy_price']:.2f}")

# --- 7. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera de Inversión")

# Mostrar fecha de inicio
st.caption(f"📅 Fecha de inicio de inversión: **{start_date.strftime('%d/%m/%Y')}** | 💰 Capital inicial: **€{capital:,.0f}**")

benchmark_ticker = None
if custom_benchmark.strip():
    benchmark_ticker = custom_benchmark.strip().upper()
elif BENCHMARK_OPTIONS[benchmark_selection]:
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_selection]

# Filtrar dividendos desde la fecha de compra
DIVIDEND_DATA_FILTERED = get_dividends_from_date(DIVIDEND_DATA, start_date)

tab1, tab2, tab3 = st.tabs(["📈 Rendimiento", "📅 Calendario de Dividendos", "💶 Importes en España"])

tickers = list(PORTFOLIO_CONFIG.keys())
with st.spinner('Actualizando precios de mercado...'):
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
                            benchmark_data = benchmark_data[~benchmark_data.index.duplicated(keep='first')]
                            
                            start_dt = pd.to_datetime(start_date)
                            
                            benchmark_before_start = benchmark_data[benchmark_data.index <= start_dt]
                            if len(benchmark_before_start) > 0:
                                benchmark_initial_price = float(benchmark_before_start.iloc[-1])
                            else:
                                benchmark_initial_price = float(benchmark_data.iloc[0])
                            
                            if benchmark_initial_price > 0:
                                benchmark_shares = capital / benchmark_initial_price
                                benchmark_filtered = benchmark_data[benchmark_data.index >= start_dt]
                                
                                if len(benchmark_filtered) > 0:
                                    benchmark_value_series = benchmark_filtered * benchmark_shares
                                    initial_point = pd.Series([capital], index=[start_plot_date])
                                    benchmark_value_series = pd.concat([initial_point, benchmark_value_series])
                                    benchmark_value_series = benchmark_value_series[~benchmark_value_series.index.duplicated(keep='first')]
                                    benchmark_value_series = benchmark_value_series.sort_index()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=benchmark_value_series.index, 
                                        y=benchmark_value_series.values, 
                                        name=f"{benchmark_ticker} (€{capital:,.0f} inv.)", 
                                        mode='lines',
                                        line=dict(color='#f59e0b', width=2, dash='dot'),
                                        hovertemplate=f'{benchmark_ticker}: €%{{y:,.0f}}<extra></extra>'
                                    ), row=1, col=1)
                                    
                                    benchmark_added = True
                                    benchmark_current_value = float(benchmark_value_series.iloc[-1])
                                    
                        except Exception as e:
                            st.warning(f"⚠️ Error procesando benchmark: {str(e)}")
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
            st.subheader("⚖️ Bandas (±10%)")
            
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
                
                status = "✅ OK"
                
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
                if "VENDER" in v: return 'color: #991b1b; background-color: #fee2e2; font-weight: bold;'
                if "COMPRAR" in v: return 'color: #1e40af; background-color: #dbeafe; font-weight: bold;'
                return 'color: #166534; background-color: #dcfce7; font-weight: bold;'
                
            st.dataframe(df_rb.style.applymap(style_rebal, subset=['Estado']), use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            <div style="background-color:#ffffff; padding:15px; border-radius:8px; margin-top:15px; border:1px solid #cbd5e1;">
                <span style="color:#475569; font-weight: 600;">Liquidez:</span>
                <span style="color:#0f172a; font-weight:bold; float:right; font-size:1.2rem;">{cash_leftover:.2f} €</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ Error de conexión con Yahoo Finance.")

# --- TAB 2: CALENDARIO DE DIVIDENDOS ---
with tab2:
    st.subheader("📅 Calendario de Dividendos")
    
    # Info sobre fecha de inicio
    st.info(f"📅 Mostrando dividendos desde la fecha de compra: **{start_date.strftime('%d/%m/%Y')}**")
    
    with st.expander("🔍 Estado de fuentes de datos"):
        status_data = []
        for ticker, status in SCRAPING_STATUS.items():
            status_data.append({
                'ETF': ticker,
                'ISIN': PORTFOLIO_CONFIG[ticker]['isin'],
                'Fuente': status
            })
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background-color:#dbeafe; border-left:4px solid #2563eb; padding:12px; border-radius:0 8px 8px 0; margin-bottom:15px;">
        <b>ℹ️ Info fiscal:</b> ETFs UCITS (Irlanda/Luxemburgo) sin retención en origen. España aplica <b>19%</b>.
    </div>
    """, unsafe_allow_html=True)
    
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    start_month = start_date.month
    
    dividend_calendar = []
    monthly_totals = {i: 0 for i in range(1, 13)}
    total_bruto = 0
    total_bruto_first_year = 0
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        div_months = div_data.get('dividend_months', [])
        annual_div = div_per_share * cfg['shares']
        first_year_div = div_data.get('first_year_dividend', 0) * cfg['shares']
        
        total_bruto += annual_div
        total_bruto_first_year += first_year_div
        
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
                # Marcar si el mes es después de la fecha de compra
                if i > start_month or start_date.year < datetime.now().year:
                    row[mes] = f"€{div_per_payment:.2f}"
                    monthly_totals[i] += div_per_payment
                else:
                    row[mes] = f"(€{div_per_payment:.2f})"  # Entre paréntesis = no cobrado este año
            else:
                row[mes] = "-"
        
        row['Total Anual'] = f"€{annual_div:.2f}"
        row['1er Año'] = f"€{first_year_div:.2f}"
        dividend_calendar.append(row)
    
    total_retencion = total_bruto * RETENCION_ESPANA
    total_neto = total_bruto - total_retencion
    total_retencion_first = total_bruto_first_year * RETENCION_ESPANA
    total_neto_first = total_bruto_first_year - total_retencion_first
    yield_cartera = (total_bruto / CAPITAL_INVERTIDO) * 100 if CAPITAL_INVERTIDO > 0 else 0
    
    st.markdown("### 📊 Resumen de Dividendos")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Div. Anual Bruto", f"€{total_bruto:,.2f}")
    col2.metric("💰 Div. Anual Neto", f"€{total_neto:,.2f}", f"-€{total_retencion:.2f} ret.")
    col3.metric(f"🗓️ 1er Año (desde {start_date.strftime('%m/%Y')})", f"€{total_neto_first:,.2f} neto")
    col4.metric("📊 Yield s/coste", f"{yield_cartera:.2f}%")
    
    st.markdown("---")
    
    # Tabla de dividendos
    st.markdown("### 📋 Detalle por ETF")
    
    div_detail = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        annual_div = div_per_share * cfg['shares']
        first_year_div = div_data.get('first_year_dividend', 0) * cfg['shares']
        yield_on_cost = (div_per_share / cfg['buy_price']) * 100 if cfg['buy_price'] > 0 else 0
        frequency = div_data.get('distribution_frequency', 'N/A')
        remaining_months = div_data.get('remaining_months_first_year', [])
        
        div_detail.append({
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Frecuencia': frequency,
            'Meses pago': ', '.join([meses[m-1] for m in div_data.get('dividend_months', [])]),
            'Div/Acc': f"€{div_per_share:.2f}",
            'Total Anual': f"€{annual_div:.2f}",
            f'1er Año': f"€{first_year_div:.2f}",
            'Yield': f"{yield_on_cost:.2f}%",
        })
    
    st.dataframe(pd.DataFrame(div_detail), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Calendario visual - solo meses relevantes
    st.markdown(f"### 📆 Distribución Mensual (desde {meses[start_month-1]} {start_date.year})")
    
    cols = st.columns(6)
    for i in range(12):
        mes_num = i + 1
        with cols[i % 6]:
            total_mes = monthly_totals[mes_num]
            neto_mes = total_mes * (1 - RETENCION_ESPANA)
            
            # Determinar si el mes está activo (después de la fecha de compra)
            is_future = mes_num > start_month or start_date.year < datetime.now().year
            
            if total_mes > 0 and is_future:
                bg_color = "#dcfce7"
                border_color = "#16a34a"
                icon = "💰"
            elif total_mes > 0:
                bg_color = "#fef3c7"
                border_color = "#f59e0b"
                icon = "⏳"
            else:
                bg_color = "#f1f5f9"
                border_color = "#94a3b8"
                icon = "📅"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; border:2px solid {border_color}; 
                        border-radius:8px; padding:10px; margin:3px 0; text-align:center; min-height:100px;">
                <div style="font-size:1.1rem;">{icon}</div>
                <div style="font-weight:bold; color:#1e293b; font-size:0.9rem;">{meses[i]}</div>
                <div style="color:#16a34a; font-weight:600; font-size:1rem;">€{total_mes:.2f}</div>
                <div style="color:#64748b; font-size:0.75rem;">Neto: €{neto_mes:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Resumen fiscal
    st.markdown("### 🏛️ Resumen Fiscal")
    
    fiscal_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        annual_div = div_per_share * cfg['shares']
        ret_esp = annual_div * RETENCION_ESPANA
        neto = annual_div - ret_esp
        
        fiscal_data.append({
            'ETF': ticker,
            'Bruto Anual': f"€{annual_div:.2f}",
            'Ret. 19%': f"€{ret_esp:.2f}",
            'Neto Anual': f"€{neto:.2f}",
        })
    
    fiscal_data.append({
        'ETF': '📊 TOTAL',
        'Bruto Anual': f"€{total_bruto:.2f}",
        'Ret. 19%': f"€{total_retencion:.2f}",
        'Neto Anual': f"€{total_neto:.2f}",
    })
    
    st.dataframe(pd.DataFrame(fiscal_data), use_container_width=True, hide_index=True)
    
    monthly_net_avg = total_neto / 12
    st.markdown(f"""
    <div style="background-color:#dcfce7; border:2px solid #16a34a; padding:15px; border-radius:10px; margin-top:15px; text-align:center;">
        <span style="color:#166534; font-size:1.1rem; font-weight:600;">💰 Ingreso Mensual Medio (Neto):</span>
        <span style="color:#166534; font-weight:bold; font-size:1.5rem; margin-left:10px;">€{monthly_net_avg:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 3: IMPORTES EN ESPAÑA ---
with tab3:
    st.subheader("💶 Dividendos en España")
    
    st.info(f"📅 Proyección desde: **{start_date.strftime('%d/%m/%Y')}**")
    
    st.markdown("""
    <div style="background-color:#fef3c7; border-left:4px solid #f59e0b; padding:12px; border-radius:0 8px 8px 0; margin-bottom:15px;">
        <b>🇪🇸 Fiscalidad:</b> ETFs UCITS sin retención origen (0%). España retiene <b>19%</b>. Neto = Bruto × 0.81
    </div>
    """, unsafe_allow_html=True)
    
    total_bruto_esp = 0
    total_first_year_esp = 0
    spain_data = []
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        annual_bruto = div_per_share * cfg['shares']
        first_year_bruto = div_data.get('first_year_dividend', 0) * cfg['shares']
        
        total_bruto_esp += annual_bruto
        total_first_year_esp += first_year_bruto
        
        ret_espana = annual_bruto * RETENCION_ESPANA
        neto_espana = annual_bruto - ret_espana
        first_year_neto = first_year_bruto * (1 - RETENCION_ESPANA)
        
        spain_data.append({
            'ETF': ticker,
            'Nombre': cfg['name'],
            'Domicilio': 'Irlanda' if cfg['isin'].startswith('IE') else 'P. Bajos',
            'Bruto Anual': f"€{annual_bruto:.2f}",
            'Ret. 19%': f"€{ret_espana:.2f}",
            'Neto Anual': f"€{neto_espana:.2f}",
            '1er Año Neto': f"€{first_year_neto:.2f}",
        })
    
    total_ret_esp = total_bruto_esp * RETENCION_ESPANA
    total_neto_esp = total_bruto_esp - total_ret_esp
    total_first_year_neto = total_first_year_esp * (1 - RETENCION_ESPANA)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Bruto Anual", f"€{total_bruto_esp:,.2f}")
    col2.metric("🇪🇸 Ret. 19%", f"-€{total_ret_esp:,.2f}")
    col3.metric("💰 Neto Anual", f"€{total_neto_esp:,.2f}")
    col4.metric(f"🗓️ 1er Año Neto", f"€{total_first_year_neto:,.2f}")
    
    st.markdown("---")
    
    st.markdown("### 📋 Desglose por ETF")
    st.dataframe(pd.DataFrame(spain_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown(f"### 📅 Cobros Mensuales (desde {meses[start_month-1]} {start_date.year})")
    
    monthly_spain = {i: {'bruto': 0, 'neto': 0, 'etfs': []} for i in range(1, 13)}
    
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_share = div_data.get('annual_dividend', 0)
        div_months = div_data.get('dividend_months', [])
        annual_div = div_per_share * cfg['shares']
        
        payments_per_year = len(div_months) if div_months else 1
        div_per_payment = annual_div / payments_per_year if payments_per_year > 0 else 0
        
        for month in div_months:
            # Solo contar meses después de la fecha de compra
            if month > start_month or start_date.year < datetime.now().year:
                monthly_spain[month]['bruto'] += div_per_payment
                monthly_spain[month]['neto'] += div_per_payment * (1 - RETENCION_ESPANA)
                monthly_spain[month]['etfs'].append(ticker.split('.')[0])
    
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    cols = st.columns(4)
    for i in range(12):
        mes_num = i + 1
        with cols[i % 4]:
            data = monthly_spain[mes_num]
            is_future = mes_num > start_month or start_date.year < datetime.now().year
            
            if data['bruto'] > 0 and is_future:
                bg_color = "#dcfce7"
                border_color = "#16a34a"
            elif data['bruto'] > 0:
                bg_color = "#fef3c7"
                border_color = "#f59e0b"
            else:
                bg_color = "#f1f5f9"
                border_color = "#94a3b8"
            
            etfs_str = ", ".join(data['etfs']) if data['etfs'] else "—"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; border:2px solid {border_color}; 
                        border-radius:8px; padding:12px; margin:3px 0; text-align:center; min-height:130px;">
                <div style="font-weight:bold; color:#1e293b; font-size:0.9rem; margin-bottom:5px;">{meses_nombres[i]}</div>
                <div style="color:#64748b; font-size:0.7rem;">Bruto: €{data['bruto']:.2f}</div>
                <div style="color:#dc2626; font-size:0.7rem;">-€{data['bruto'] * RETENCION_ESPANA:.2f}</div>
                <div style="color:#16a34a; font-weight:700; font-size:1.1rem; margin-top:3px;">€{data['neto']:.2f}</div>
                <div style="color:#64748b; font-size:0.65rem; margin-top:3px;">{etfs_str}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Resumen
    avg_mensual_neto = total_neto_esp / 12
    yield_neto = (total_neto_esp / CAPITAL_INVERTIDO) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color:#fff; border:2px solid #2563eb; padding:15px; border-radius:10px;">
            <h4 style="color:#1e293b; margin-bottom:10px;">📊 Totales Anuales</h4>
            <table style="width:100%; color:#334155; font-size:0.9rem;">
                <tr><td>Dividendos Brutos:</td><td style="text-align:right; font-weight:bold;">€{total_bruto_esp:,.2f}</td></tr>
                <tr><td>Retención 19%:</td><td style="text-align:right; color:#dc2626;">-€{total_ret_esp:,.2f}</td></tr>
                <tr style="border-top:1px solid #e2e8f0;"><td style="padding-top:8px;"><b>NETO:</b></td><td style="text-align:right; font-weight:bold; color:#16a34a; padding-top:8px;">€{total_neto_esp:,.2f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color:#fff; border:2px solid #16a34a; padding:15px; border-radius:10px;">
            <h4 style="color:#1e293b; margin-bottom:10px;">📅 Promedios</h4>
            <table style="width:100%; color:#334155; font-size:0.9rem;">
                <tr><td>Media Mensual Neto:</td><td style="text-align:right; font-weight:bold; color:#16a34a;">€{avg_mensual_neto:,.2f}</td></tr>
                <tr><td>Yield Bruto:</td><td style="text-align:right;">{(total_bruto_esp/CAPITAL_INVERTIDO)*100:.2f}%</td></tr>
                <tr><td>Yield Neto:</td><td style="text-align:right; font-weight:bold;">{yield_neto:.2f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#eff6ff; border-left:4px solid #3b82f6; padding:12px; border-radius:0 8px 8px 0; margin-top:15px; font-size:0.85rem;">
        <b>📝 Notas:</b><br>
        • Importes estimados basados en dividendos históricos<br>
        • El 19% es a cuenta del IRPF (se regulariza en la declaración)<br>
        • ETFs de Países Bajos pueden tener retención del 15% recuperable
    </div>
    """, unsafe_allow_html=True)
