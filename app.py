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
import time

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Cartera Permanente Pro", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc !important; }
    .main h1, .main h2, .main h3 { color: #1e293b !important; font-family: 'Segoe UI', sans-serif; font-weight: 800; }
    .main p, .main li, .main div { color: #334155; }
    section[data-testid="stSidebar"] { min-width: 320px !important; width: 320px !important; background-color: #0f172a !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div { color: #f8fafc !important; }
    div[data-baseweb="input"], div[data-baseweb="base-input"] { background-color: #1e293b !important; border: 1px solid #475569 !important; border-radius: 6px !important; }
    input[class*="st-"] { color: #ffffff !important; font-size: 1rem !important; }
    div[data-baseweb="select"] { background-color: #1e293b !important; }
    div[data-baseweb="select"] > div { background-color: #1e293b !important; border: 1px solid #475569 !important; border-radius: 6px !important; }
    div[data-baseweb="select"] svg, div[data-testid="stDateInput"] svg { fill: white !important; }
    div[data-testid="stDateInput"] > div > div { background-color: #1e293b !important; border: 1px solid #475569 !important; border-radius: 6px !important; }
    div[data-testid="stNumberInput"] > div > div > input { background-color: #1e293b !important; color: #ffffff !important; border: 1px solid #475569 !important; border-radius: 6px !important; }
    section[data-testid="stSidebar"] button { background-color: #2563eb !important; color: white !important; font-weight: bold; border: none !important; border-radius: 6px !important; padding: 0.5rem 1rem !important; }
    section[data-testid="stSidebar"] button:hover { background-color: #1d4ed8 !important; }
    section[data-testid="stMain"] div[data-testid="stMetric"] { background-color: #ffffff !important; border: 1px solid #e2e8f0; border-left: 6px solid #2563eb; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    section[data-testid="stMain"] div[data-testid="stMetricValue"] div { font-size: 2rem !important; color: #0f172a !important; font-weight: 800 !important; }
    section[data-testid="stMain"] div[data-testid="stMetricLabel"] p { font-size: 1rem !important; color: #64748b !important; font-weight: 600 !important; }
    section[data-testid="stMain"] div[data-testid="stMetricDelta"] div { font-size: 1.1rem !important; font-weight: 700 !important; }
    .stDataFrame { border: 1px solid #cbd5e1; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #e2e8f0; border-radius: 6px 6px 0 0; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: white !important; }
    .streamlit-expanderHeader { background-color: #f1f5f9; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONFIGURACIÓN DE LA CARTERA ---
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

DIGRIN_TICKER_MAP = {
    'IQQM.DE': 'IQQM.DE',
    'TDIV.AS': 'TDIV.AS',
    'EHDV.DE': 'EHDV.DE',
    'IUSM.DE': 'IUSM.DE',
    'JNKE.MI': 'JNKE.L',
}

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

# --- 3. SCRAPING DIGRIN ---
@st.cache_data(ttl=86400, show_spinner=False)
def scrape_digrin_dividends(ticker, digrin_ticker):
    url = f"https://www.digrin.com/stocks/detail/{digrin_ticker}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    dividend_data = {
        'annual_dividend': 0, 'dividend_yield': 0, 'distribution_frequency': '',
        'last_dividend': 0, 'last_ex_date': '', 'dividend_months': [],
        'history': [], 'upcoming': [], 'source': 'digrin.com'
    }
    try:
        session = requests.Session()
        session.get("https://www.digrin.com/", headers=headers, timeout=10)
        time.sleep(0.5)
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='table-striped')
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                header_row = t.find('tr')
                if header_row:
                    header_text = header_row.get_text()
                    if 'Ex-dividend' in header_text or 'Dividend amount' in header_text:
                        table = t
                        break
        if not table:
            dividend_data['source'] = f'digrin.com - tabla no encontrada'
            return dividend_data
        tbody = table.find('tbody')
        rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]
        if not rows:
            dividend_data['source'] = 'digrin.com - sin filas'
            return dividend_data
        today = datetime.now()
        one_year_ago = today - timedelta(days=400)
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                try:
                    ex_date_text = cells[0].get_text(strip=True)
                    ex_date = None
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y', '%m/%d/%Y']:
                        try:
                            ex_date = datetime.strptime(ex_date_text, fmt)
                            break
                        except:
                            continue
                    if not ex_date:
                        continue
                    pay_date_text = cells[1].get_text(strip=True)
                    pay_date = None
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y', '%m/%d/%Y']:
                        try:
                            pay_date = datetime.strptime(pay_date_text, fmt)
                            break
                        except:
                            continue
                    div_cell_text = cells[2].get_text(strip=True)
                    match = re.match(r'([\d.,]+)\s*([A-Z]{3})', div_cell_text)
                    if match:
                        amount_str = match.group(1).replace(',', '.')
                        if amount_str.count('.') > 1:
                            parts = amount_str.rsplit('.', 1)
                            amount_str = parts[0].replace('.', '') + '.' + parts[1]
                        amount = float(amount_str)
                        currency = match.group(2)
                        div_entry = {
                            'ex_date': ex_date, 'pay_date': pay_date,
                            'amount': amount, 'currency': currency, 'month': ex_date.month
                        }
                        dividend_data['history'].append(div_entry)
                        if ex_date > today:
                            dividend_data['upcoming'].append(div_entry)
                except Exception:
                    continue
        if dividend_data['history']:
            dividend_data['history'].sort(key=lambda x: x['ex_date'], reverse=True)
            dividend_data['upcoming'].sort(key=lambda x: x['ex_date'])
            dividend_data['last_dividend'] = dividend_data['history'][0]['amount']
            dividend_data['last_ex_date'] = dividend_data['history'][0]['ex_date'].strftime('%Y-%m-%d')
            recent_dividends = [d for d in dividend_data['history']
                                if one_year_ago <= d['ex_date'] <= today]
            if recent_dividends:
                dividend_data['annual_dividend'] = sum(d['amount'] for d in recent_dividends)
                payment_months = sorted(list(set(d['month'] for d in recent_dividends)))
                dividend_data['dividend_months'] = payment_months
                num_payments = len(recent_dividends)
                if num_payments >= 10:
                    dividend_data['distribution_frequency'] = 'Mensual'
                elif num_payments >= 4:
                    dividend_data['distribution_frequency'] = 'Trimestral'
                elif num_payments >= 2:
                    dividend_data['distribution_frequency'] = 'Semestral'
                else:
                    dividend_data['distribution_frequency'] = 'Anual'
            dividend_data['source'] = f'digrin.com ✓ ({len(dividend_data["history"])} registros)'
        else:
            dividend_data['source'] = 'digrin.com - sin historial parseado'
        return dividend_data
    except requests.exceptions.HTTPError as e:
        dividend_data['source'] = f'Error HTTP: {e.response.status_code}'
        return dividend_data
    except Exception as e:
        dividend_data['source'] = f'Error: {str(e)[:40]}'
        return dividend_data


# --- DATOS DE RESPALDO ---
FALLBACK_DIVIDENDS = {
    'IQQM.DE': {
        'annual_dividend': 2.26,
        'dividend_yield': 2.89,
        'distribution_frequency': 'Trimestral',
        'dividend_months': [3, 6, 9, 12],
        'last_dividend': 0.2158,
        'last_ex_date': '2025-12-11',
        'history': [
            {'ex_date': datetime(2025, 12, 11), 'amount': 0.2158, 'currency': 'EUR', 'month': 12},
            {'ex_date': datetime(2025, 9, 11), 'amount': 0.5946, 'currency': 'EUR', 'month': 9},
            {'ex_date': datetime(2025, 6, 12), 'amount': 1.6865, 'currency': 'EUR', 'month': 6},
            {'ex_date': datetime(2025, 3, 13), 'amount': 0.0748, 'currency': 'EUR', 'month': 3},
            {'ex_date': datetime(2024, 12, 12), 'amount': 0.3523, 'currency': 'EUR', 'month': 12},
            {'ex_date': datetime(2024, 9, 12), 'amount': 0.5778, 'currency': 'EUR', 'month': 9},
            {'ex_date': datetime(2024, 6, 13), 'amount': 1.2545, 'currency': 'EUR', 'month': 6},
            {'ex_date': datetime(2024, 3, 14), 'amount': 0.0726, 'currency': 'EUR', 'month': 3},
        ],
        'upcoming': [],
        'source': 'Fallback digrin.com'
    },
    'TDIV.AS': {
        'annual_dividend': 1.85,
        'dividend_yield': 3.96,
        'distribution_frequency': 'Trimestral',
        'dividend_months': [3, 6, 9, 12],
        'last_dividend': 0.47,
        'last_ex_date': '2024-12-16',
        'history': [
            {'ex_date': datetime(2024, 12, 16), 'amount': 0.47, 'currency': 'EUR', 'month': 12},
            {'ex_date': datetime(2024, 9, 16), 'amount': 0.46, 'currency': 'EUR', 'month': 9},
            {'ex_date': datetime(2024, 6, 17), 'amount': 0.45, 'currency': 'EUR', 'month': 6},
            {'ex_date': datetime(2024, 3, 18), 'amount': 0.47, 'currency': 'EUR', 'month': 3},
        ],
        'upcoming': [],
        'source': 'Fallback'
    },
    'EHDV.DE': {
        'annual_dividend': 1.48,
        'dividend_yield': 4.69,
        'distribution_frequency': 'Trimestral',
        'dividend_months': [1, 4, 7, 10],
        'last_dividend': 0.37,
        'last_ex_date': '2024-10-17',
        'history': [
            {'ex_date': datetime(2024, 10, 17), 'amount': 0.37, 'currency': 'EUR', 'month': 10},
            {'ex_date': datetime(2024, 7, 18), 'amount': 0.38, 'currency': 'EUR', 'month': 7},
            {'ex_date': datetime(2024, 4, 18), 'amount': 0.36, 'currency': 'EUR', 'month': 4},
            {'ex_date': datetime(2024, 1, 18), 'amount': 0.37, 'currency': 'EUR', 'month': 1},
        ],
        'upcoming': [],
        'source': 'Fallback'
    },
    'IUSM.DE': {
        'annual_dividend': 5.20,
        'dividend_yield': 3.44,
        'distribution_frequency': 'Semestral',
        'dividend_months': [4, 10],
        'last_dividend': 2.65,
        'last_ex_date': '2024-10-10',
        'history': [
            {'ex_date': datetime(2024, 10, 10), 'amount': 2.65, 'currency': 'EUR', 'month': 10},
            {'ex_date': datetime(2024, 4, 11), 'amount': 2.55, 'currency': 'EUR', 'month': 4},
            {'ex_date': datetime(2023, 10, 12), 'amount': 2.48, 'currency': 'EUR', 'month': 10},
            {'ex_date': datetime(2023, 4, 13), 'amount': 2.40, 'currency': 'EUR', 'month': 4},
        ],
        'upcoming': [],
        'source': 'Fallback'
    },
    'JNKE.MI': {
        'annual_dividend': 2.72,
        'dividend_yield': 5.22,
        'distribution_frequency': 'Semestral',
        'dividend_months': [6, 12],
        'last_dividend': 1.38,
        'last_ex_date': '2024-12-12',
        'history': [
            {'ex_date': datetime(2024, 12, 12), 'amount': 1.38, 'currency': 'EUR', 'month': 12},
            {'ex_date': datetime(2024, 6, 13), 'amount': 1.34, 'currency': 'EUR', 'month': 6},
            {'ex_date': datetime(2023, 12, 14), 'amount': 1.32, 'currency': 'EUR', 'month': 12},
            {'ex_date': datetime(2023, 6, 15), 'amount': 1.28, 'currency': 'EUR', 'month': 6},
        ],
        'upcoming': [],
        'source': 'Fallback'
    }
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_dividend_data():
    dividend_data = {}
    scraping_status = {}
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        digrin_ticker = DIGRIN_TICKER_MAP.get(ticker, ticker)
        scraped_data = scrape_digrin_dividends(ticker, digrin_ticker)
        if scraped_data.get('annual_dividend', 0) > 0 and len(scraped_data.get('history', [])) > 0:
            dividend_data[ticker] = scraped_data
            scraping_status[ticker] = scraped_data.get('source', 'digrin.com')
        else:
            fallback = FALLBACK_DIVIDENDS.get(ticker, {})
            if fallback:
                dividend_data[ticker] = fallback.copy()
                scraping_status[ticker] = f"Fallback - {scraped_data.get('source', 'scraping falló')}"
            else:
                dividend_data[ticker] = {
                    'annual_dividend': 0, 'dividend_yield': 0,
                    'distribution_frequency': 'Desconocida',
                    'dividend_months': [], 'history': [], 'upcoming': [], 'source': 'Sin datos'
                }
                scraping_status[ticker] = 'Sin datos disponibles'
    return dividend_data, scraping_status


def get_dividends_from_date(dividend_data, start_date):
    """
    Calcula dividendos proyectados desde la fecha de inicio.
    Usa el histórico real para calcular div_per_payment por cada mes de pago.
    """
    filtered_data = {}
    start_dt = datetime.combine(start_date, datetime.min.time())
    today = datetime.now()

    for ticker, data in dividend_data.items():
        filtered = data.copy()
        div_months = data.get('dividend_months', [])
        history = data.get('history', [])
        annual_div = data.get('annual_dividend', 0)

        # Dividendos históricos después de la fecha de compra
        future_dividends = [
            div for div in history
            if (div['ex_date'] if isinstance(div['ex_date'], datetime)
                else datetime.combine(div['ex_date'], datetime.min.time())) > start_dt
        ]

        if div_months:
            payments_per_year = len(div_months)

            # Calcular pago promedio usando últimos pagos del historial
            if history and len(history) >= payments_per_year:
                recent_payments = history[:payments_per_year]
                div_per_payment = sum(d['amount'] for d in recent_payments) / len(recent_payments)
            else:
                div_per_payment = annual_div / payments_per_year if payments_per_year > 0 else 0

            # ── CLAVE: primer año = pagos cuya fecha de ex-div cae DESPUÉS de start_date ──
            # Usamos el historial real si hay datos post-compra, si no proyectamos
            if future_dividends:
                first_year_dividend = sum(d['amount'] for d in future_dividends
                                          if d['ex_date'] <= today + timedelta(days=365))
            else:
                # Proyección: meses de pago que quedan desde el mes de inicio
                # Para una cartera permanente mostramos TODOS los meses pero
                # el primer año solo cuenta desde start_date
                remaining_months = [m for m in div_months if m >= start_date.month]
                first_year_dividend = div_per_payment * len(remaining_months)

            filtered['first_year_dividend'] = first_year_dividend
            filtered['first_year_payments'] = len([m for m in div_months if m >= start_date.month])
            filtered['div_per_payment'] = div_per_payment
            filtered['future_dividends'] = future_dividends
        else:
            filtered['first_year_dividend'] = 0
            filtered['first_year_payments'] = 0
            filtered['div_per_payment'] = 0
            filtered['future_dividends'] = []

        filtered_data[ticker] = filtered
    return filtered_data


def build_dividend_cashflow(dividend_data, portfolio_config, start_date):
    """
    Construye una serie temporal de flujos de caja de dividendos
    (brutos y netos) para la cartera completa, desde start_date.
    Se usa para sumar al valor de la cartera en el gráfico.
    """
    start_dt = datetime.combine(start_date, datetime.min.time())
    cashflows = {}  # fecha -> importe neto acumulado

    for ticker, data in dividend_data.items():
        cfg = portfolio_config[ticker]
        history = data.get('history', [])
        for div in history:
            ex_date = div['ex_date'] if isinstance(div['ex_date'], datetime) \
                else datetime.combine(div['ex_date'], datetime.min.time())
            if ex_date >= start_dt:
                date_key = ex_date.date()
                importe = div['amount'] * cfg['shares']
                cashflows[date_key] = cashflows.get(date_key, 0) + importe

    if not cashflows:
        return pd.Series(dtype=float)

    cf_series = pd.Series(cashflows).sort_index()
    cf_series.index = pd.to_datetime(cf_series.index)
    return cf_series


# --- 4. FUNCIONES DE MERCADO ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data_cached(tickers):
    start_date_dl = datetime.now() - timedelta(days=365 * 5)
    try:
        data = yf.download(tickers, start=start_date_dl, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'] if 'Close' in data.columns.get_level_values(0) else data.iloc[:, :len(tickers)]
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
            series = data['Close'].iloc[:, 0] if 'Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
        elif 'Close' in data.columns:
            series = data['Close']
        else:
            series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series
    except Exception:
        return pd.Series(dtype=float)


def calculate_metrics(series, capital_inicial):
    if series.empty or len(series) < 2:
        return 0.0, 0.0, 0.0, pd.Series(0, index=series.index)
    ret = series.pct_change().fillna(0)
    days = (series.index[-1] - series.index[0]).days
    current_val = series.iloc[-1]
    if days > 0:
        total_ret = (current_val / capital_inicial) - 1
        cagr = (1 + total_ret) ** (365.25 / days) - 1 if total_ret > -0.9 else 0
    else:
        cagr = 0.0
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    rf = 0.03
    sharpe = np.sqrt(252) * (ret - rf / 252).mean() / ret.std() if ret.std() > 0 else 0.0
    return cagr, max_dd, sharpe, dd


# --- 5. CARGAR DATOS ---
with st.spinner('Cargando datos de dividendos desde digrin.com...'):
    DIVIDEND_DATA, SCRAPING_STATUS = get_all_dividend_data()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    capital = st.number_input("Capital Inicial (€)", value=13000, step=500)
    start_date = st.date_input("Fecha Inicio Inversión", value=FECHA_COMPRA,
                               help="Fecha de compra de las acciones")
    st.markdown("---")
    st.subheader("📊 Benchmark")
    benchmark_selection = st.selectbox("Seleccionar índice:", options=list(BENCHMARK_OPTIONS.keys()), index=0)
    custom_benchmark = st.text_input("Ticker personalizado:", placeholder="Ej: VWCE.DE, ^GSPC")
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
        st.caption(f"• {ticker}: {cfg['shares']} × €{cfg['buy_price']:.2f}")

# --- 7. LÓGICA PRINCIPAL ---
st.title("Dashboard de Cartera de Inversión")
st.caption(f"📅 Fecha de inicio: **{start_date.strftime('%d/%m/%Y')}** | 💰 Capital: **€{capital:,.0f}**")

benchmark_ticker = None
if custom_benchmark.strip():
    benchmark_ticker = custom_benchmark.strip().upper()
elif BENCHMARK_OPTIONS[benchmark_selection]:
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_selection]

DIVIDEND_DATA_FILTERED = get_dividends_from_date(DIVIDEND_DATA, start_date)

# Cashflow de dividendos históricos (para sumar al gráfico de rendimiento)
DIV_CASHFLOW = build_dividend_cashflow(DIVIDEND_DATA, PORTFOLIO_CONFIG, start_date)

tab1, tab2, tab3 = st.tabs(["📈 Rendimiento", "📅 Calendario de Dividendos", "💶 Importes en España"])

tickers = list(PORTFOLIO_CONFIG.keys())
with st.spinner('Actualizando precios de mercado...'):
    full_df = get_market_data_cached(tickers)

# ══════════════════════════════════════════════════════
# TAB 1: RENDIMIENTO
# ══════════════════════════════════════════════════════
with tab1:
    if not full_df.empty:
        full_df.index = pd.to_datetime(full_df.index)
        df_analysis = full_df[full_df.index >= pd.to_datetime(start_date)].copy()
        df_analysis = df_analysis.ffill().dropna()

        if len(df_analysis) == 0:
            last_known = full_df.ffill().iloc[-1]
            df_analysis = pd.DataFrame([last_known], index=[pd.to_datetime(start_date)])

        latest_prices = df_analysis.ffill().iloc[-1]

        # ── Valor de mercado de la cartera ──
        portfolio_series = pd.DataFrame(index=df_analysis.index)
        portfolio_series['Mercado'] = 0
        portfolio_shares = {}
        invested_cash = 0

        for t in tickers:
            n_shares = PORTFOLIO_CONFIG[t]['shares']
            buy_price = PORTFOLIO_CONFIG[t]['buy_price']
            portfolio_shares[t] = n_shares
            invested_cash += n_shares * buy_price
            portfolio_series['Mercado'] += df_analysis[t] * n_shares

        cash_leftover = capital - invested_cash
        portfolio_series['Mercado'] += cash_leftover

        # ── Sumar dividendos cobrados acumulados (Total Return) ──
        if not DIV_CASHFLOW.empty:
            # Reindexar al calendario de trading y acumular
            div_cum = DIV_CASHFLOW.reindex(df_analysis.index, fill_value=0).cumsum()
            portfolio_series['Total'] = portfolio_series['Mercado'] + div_cum
        else:
            portfolio_series['Total'] = portfolio_series['Mercado'].copy()

        current_total = portfolio_series['Total'].iloc[-1]
        cagr_real, max_dd_real, sharpe_real, dd_series = calculate_metrics(
            portfolio_series['Total'], capital
        )
        abs_ret = current_total - capital
        pct_ret = (current_total / capital) - 1

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valor Actual (TR)", f"€{current_total:,.0f}", f"Inv: €{capital:,.0f}", delta_color="off")
        k2.metric(f"Rentabilidad Total Return", f"{pct_ret:+.2%}",
                  f"{abs_ret:+,.0f} € | CAGR {cagr_real:.1%}")
        k3.metric("Drawdown Actual", f"{dd_series.iloc[-1]:.2%}",
                  f"Máx: {max_dd_real:.2%}", delta_color="inverse")
        k4.metric("Ratio Sharpe", f"{sharpe_real:.2f}")

        st.markdown("---")

        col_graph, col_table = st.columns([2, 1])

        with col_graph:
            title_bench = f"📈 Evolución Total Return: Mi Cartera vs {benchmark_ticker}" \
                if benchmark_ticker else "📈 Evolución Total Return"
            st.subheader(title_bench)

            start_plot_date = portfolio_series.index[0] - timedelta(days=1)

            row_init_val = pd.DataFrame({'Total': [capital]}, index=[start_plot_date])
            plot_series_val = pd.concat([row_init_val, portfolio_series[['Total']]]).sort_index()

            row_init_dd = pd.Series([0.0], index=[start_plot_date])
            plot_series_dd = pd.concat([row_init_dd, dd_series]).sort_index()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.05)

            fig.add_trace(go.Scatter(
                x=plot_series_val.index, y=plot_series_val['Total'],
                name="Mi Cartera (TR)", mode='lines',
                line=dict(color='#2563eb', width=3),
                hovertemplate='Cartera TR: €%{y:,.0f}<extra></extra>'
            ), row=1, col=1)

            # También mostrar valor de mercado (sin dividendos) como referencia
            row_init_mkt = pd.DataFrame({'Mercado': [capital]}, index=[start_plot_date])
            plot_mkt = pd.concat([row_init_mkt, portfolio_series[['Mercado']]]).sort_index()
            fig.add_trace(go.Scatter(
                x=plot_mkt.index, y=plot_mkt['Mercado'],
                name="Mi Cartera (precio)", mode='lines',
                line=dict(color='#93c5fd', width=1.5, dash='dash'),
                hovertemplate='Cartera precio: €%{y:,.0f}<extra></extra>'
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

                        bm_before = benchmark_data[benchmark_data.index <= start_dt]
                        benchmark_initial_price = float(bm_before.iloc[-1]) if len(bm_before) > 0 \
                            else float(benchmark_data.iloc[0])

                        if benchmark_initial_price > 0:
                            bm_shares = capital / benchmark_initial_price
                            bm_filtered = benchmark_data[benchmark_data.index >= start_dt]

                            if len(bm_filtered) > 0:
                                bm_value = bm_filtered * bm_shares
                                init_pt = pd.Series([capital], index=[start_plot_date])
                                bm_value = pd.concat([init_pt, bm_value])
                                bm_value = bm_value[~bm_value.index.duplicated(keep='first')].sort_index()

                                fig.add_trace(go.Scatter(
                                    x=bm_value.index, y=bm_value.values,
                                    name=f"{benchmark_ticker} (€{capital:,.0f})",
                                    mode='lines',
                                    line=dict(color='#f59e0b', width=2, dash='dot'),
                                    hovertemplate=f'{benchmark_ticker}: €%{{y:,.0f}}<extra></extra>'
                                ), row=1, col=1)

                                benchmark_added = True
                                benchmark_current_value = float(bm_value.iloc[-1])
                    except Exception as e:
                        st.warning(f"⚠️ Error procesando benchmark: {str(e)}")
                else:
                    st.warning(f"⚠️ No se pudieron obtener datos para {benchmark_ticker}")

            fig.add_trace(go.Scatter(
                x=plot_series_dd.index, y=plot_series_dd,
                name="Drawdown", mode='lines',
                line=dict(color='#dc2626', width=1),
                fill='tozeroy', fillcolor='rgba(220,38,38,0.1)',
                hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
            ), row=2, col=1)

            fig.update_layout(
                height=540, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), font=dict(color='#334155')
            )
            fig.update_yaxes(title_text="Valor (€)", gridcolor='#e2e8f0', row=1, col=1, tickformat=",")
            fig.update_yaxes(title_text="DD", tickformat=".0%", gridcolor='#e2e8f0', row=2, col=1)
            fig.update_xaxes(gridcolor='#e2e8f0')

            st.plotly_chart(fig, use_container_width=True)

            if benchmark_ticker and benchmark_added and benchmark_current_value is not None:
                portfolio_ret = (float(current_total) - capital) / capital
                benchmark_ret = (benchmark_current_value - capital) / capital
                diff_euros = float(current_total) - benchmark_current_value
                outperformance = portfolio_ret - benchmark_ret

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                col_p1.metric("📊 Mi Cartera (TR)", f"€{current_total:,.0f}", f"{portfolio_ret:+.2%}")
                col_p2.metric(f"📈 {benchmark_ticker}", f"€{benchmark_current_value:,.0f}", f"{benchmark_ret:+.2%}")
                if diff_euros >= 0:
                    col_p3.metric("💰 Diferencia", f"+€{diff_euros:,.0f}")
                    col_p4.metric("🏆 Outperformance", f"+{outperformance:.2%}")
                else:
                    col_p3.metric("💸 Diferencia", f"€{diff_euros:,.0f}")
                    col_p4.metric("📉 Underperformance", f"{outperformance:.2%}")

        with col_table:
            st.subheader("⚖️ Bandas de Rebalanceo (±10%)")
            rebal_data = []
            BAND_ABS = 0.10
            for t in tickers:
                target = PORTFOLIO_CONFIG[t]['target']
                n_shares = portfolio_shares[t]
                p_now = float(latest_prices[t]) if not pd.isna(latest_prices[t]) else 0.0
                val_act = n_shares * p_now
                w_real = val_act / portfolio_series['Mercado'].iloc[-1] \
                    if portfolio_series['Mercado'].iloc[-1] > 0 else 0
                status = "✅ OK"
                if w_real > target + BAND_ABS:
                    status = "🔴 VENDER"
                elif w_real < max(0, target - BAND_ABS):
                    status = "🔵 COMPRAR"
                rebal_data.append({
                    "Ticker": t, "Acc.": n_shares,
                    "Valor": f"€{val_act:,.0f}", "Peso": f"{w_real:.1%}", "Estado": status
                })

            df_rb = pd.DataFrame(rebal_data)

            def style_rebal(v):
                if "VENDER" in str(v): return 'color:#991b1b;background-color:#fee2e2;font-weight:bold;'
                if "COMPRAR" in str(v): return 'color:#1e40af;background-color:#dbeafe;font-weight:bold;'
                return 'color:#166534;background-color:#dcfce7;font-weight:bold;'

            st.dataframe(df_rb.style.map(style_rebal, subset=['Estado']),
                         use_container_width=True, hide_index=True)

            dividendos_cobrados = DIV_CASHFLOW.sum() if not DIV_CASHFLOW.empty else 0
            st.markdown(f"""
            <div style="background-color:#fff;padding:12px;border-radius:8px;
                        margin-top:12px;border:1px solid #cbd5e1;">
                <div style="color:#475569;font-weight:600;font-size:0.85rem;">Liquidez no invertida</div>
                <div style="color:#0f172a;font-weight:bold;font-size:1.2rem;">€{cash_leftover:.2f}</div>
                <div style="color:#475569;font-weight:600;font-size:0.85rem;margin-top:8px;">Dividendos cobrados (bruto)</div>
                <div style="color:#16a34a;font-weight:bold;font-size:1.2rem;">€{dividendos_cobrados:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Error de conexión con Yahoo Finance.")

# ══════════════════════════════════════════════════════
# TAB 2: CALENDARIO DE DIVIDENDOS
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("📅 Calendario de Dividendos")
    st.info(f"📅 Dividendos desde la fecha de compra: **{start_date.strftime('%d/%m/%Y')}**")

    with st.expander("🔍 Estado de fuentes de datos (digrin.com)", expanded=False):
        status_data = []
        for ticker, status in SCRAPING_STATUS.items():
            dv = DIVIDEND_DATA.get(ticker, {})
            status_data.append({
                'ETF': ticker,
                'Digrin ID': DIGRIN_TICKER_MAP.get(ticker, ticker),
                'Fuente': status,
                'Último Div.': f"€{dv.get('last_dividend', 0):.4f}",
                'Fecha': dv.get('last_ex_date', 'N/A'),
                'Registros': len(dv.get('history', []))
            })
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background-color:#dbeafe;border-left:4px solid #2563eb;padding:12px;
                border-radius:0 8px 8px 0;margin-bottom:15px;">
        <b>ℹ️ Info fiscal:</b> ETFs UCITS sin retención en origen. España aplica <b>19%</b>.
    </div>""", unsafe_allow_html=True)

    MESES = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    start_month = start_date.month
    today_month = datetime.now().month
    today_year  = datetime.now().year
    start_year  = start_date.year

    # ── Calcular totales ──
    total_bruto = 0
    total_bruto_first_year = 0

    # monthly_totals: importe bruto total de la cartera por mes (ciclo anual completo)
    monthly_totals = {i: 0.0 for i in range(1, 13)}

    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        annual_div   = div_data.get('annual_dividend', 0) * cfg['shares']
        first_yr_div = div_data.get('first_year_dividend', 0) * cfg['shares']
        div_per_payment = div_data.get('div_per_payment', 0)
        div_months   = div_data.get('dividend_months', [])

        total_bruto            += annual_div
        total_bruto_first_year += first_yr_div

        payments_per_year = len(div_months) if div_months else 1
        if div_per_payment == 0:
            div_per_payment = (div_data.get('annual_dividend', 0) / payments_per_year
                               if payments_per_year > 0 else 0)

        payment_total = div_per_payment * cfg['shares']

        # ── FIX PRINCIPAL: poblar TODOS los meses de pago (ciclo anual) ──
        for month in div_months:
            monthly_totals[month] += payment_total

    total_retencion = total_bruto * RETENCION_ESPANA
    total_neto = total_bruto - total_retencion
    total_ret_first = total_bruto_first_year * RETENCION_ESPANA
    total_neto_first = total_bruto_first_year - total_ret_first
    yield_cartera = (total_bruto / CAPITAL_INVERTIDO) * 100 if CAPITAL_INVERTIDO > 0 else 0

    # ── Métricas resumen ──
    st.markdown("### 📊 Resumen de Dividendos")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Div. Anual Bruto",  f"€{total_bruto:,.2f}")
    col2.metric("💰 Div. Anual Neto",   f"€{total_neto:,.2f}", f"-€{total_retencion:.2f} ret.")
    col3.metric(f"🗓️ 1er Año (desde {start_date.strftime('%m/%Y')})", f"€{total_neto_first:,.2f} neto")
    col4.metric("📊 Yield s/coste",     f"{yield_cartera:.2f}%")

    st.markdown("---")

    # ── Historial reciente ──
    st.markdown("### 📜 Historial Reciente de Dividendos")
    history_rows = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        for div in DIVIDEND_DATA.get(ticker, {}).get('history', [])[:6]:
            ex_date = div['ex_date']
            date_str = ex_date.strftime('%Y-%m-%d') if isinstance(ex_date, datetime) else str(ex_date)
            history_rows.append({
                'ETF': ticker,
                'Fecha Ex-Div': date_str,
                'Importe/Acc': f"€{div['amount']:.4f}",
                'Total Bruto': f"€{div['amount'] * cfg['shares']:.2f}",
                'Neto España': f"€{div['amount'] * cfg['shares'] * (1 - RETENCION_ESPANA):.2f}"
            })

    if history_rows:
        df_hist = pd.DataFrame(history_rows).sort_values('Fecha Ex-Div', ascending=False)
        st.dataframe(df_hist.head(25), use_container_width=True, hide_index=True)
    else:
        st.warning("No hay historial de dividendos disponible")

    st.markdown("---")

    # ── Detalle por ETF ──
    st.markdown("### 📋 Detalle por ETF")
    div_detail = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data  = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_pa    = div_data.get('annual_dividend', 0)
        annual_div  = div_pa * cfg['shares']
        first_yr_div = div_data.get('first_year_dividend', 0) * cfg['shares']
        yoc = (div_pa / cfg['buy_price']) * 100 if cfg['buy_price'] > 0 else 0
        div_months = div_data.get('dividend_months', [])
        div_detail.append({
            'ETF': ticker,
            'Nombre': cfg['name'][:25],
            'Frecuencia': div_data.get('distribution_frequency', 'N/A'),
            'Meses pago': ', '.join([MESES[m-1] for m in div_months]) if div_months else 'N/A',
            'Div/Acc Anual': f"€{div_pa:.4f}",
            'Total Anual': f"€{annual_div:.2f}",
            '1er Año Bruto': f"€{div_data.get('first_year_dividend',0)*cfg['shares']:.2f}",
            'Yield YoC': f"{yoc:.2f}%",
        })
    st.dataframe(pd.DataFrame(div_detail), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Calendario visual ANUAL (todos los meses) ──
    st.markdown("### 📆 Distribución Mensual Anual")
    st.caption("Se muestra el ciclo anual completo. Verde = mes futuro, Amarillo = mes pasado este año.")

    cols = st.columns(6)
    for i in range(12):
        mes_num = i + 1
        with cols[i % 6]:
            total_mes = monthly_totals[mes_num]
            neto_mes  = total_mes * (1 - RETENCION_ESPANA)

            # Colorear según si el mes ya pasó este año o es futuro
            if today_year > start_year:
                # Cartera ya tiene más de un año
                es_futuro = mes_num >= today_month
            else:
                # Primer año de cartera
                es_futuro = mes_num >= today_month
            es_desde_inicio = mes_num >= start_month if today_year == start_year else True

            if total_mes > 0 and es_futuro:
                bg, border, icon = "#dcfce7", "#16a34a", "💰"
            elif total_mes > 0:
                bg, border, icon = "#fef3c7", "#f59e0b", "⏳"
            else:
                bg, border, icon = "#f1f5f9", "#94a3b8", "📅"

            st.markdown(f"""
            <div style="background-color:{bg};border:2px solid {border};border-radius:8px;
                        padding:10px;margin:3px 0;text-align:center;min-height:105px;">
                <div style="font-size:1rem;">{icon}</div>
                <div style="font-weight:bold;color:#1e293b;font-size:0.9rem;">{MESES[i]}</div>
                <div style="color:#16a34a;font-weight:600;font-size:0.95rem;">€{total_mes:.2f}</div>
                <div style="color:#64748b;font-size:0.75rem;">Neto: €{neto_mes:.2f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Proyección año a año (5 años) ──
    st.markdown("### 📈 Proyección Anual de Dividendos (5 años)")
    st.caption("Asumiendo dividendos constantes (sin crecimiento). Ajusta manualmente si lo deseas.")

    proyeccion_rows = []
    current_year = datetime.now().year
    for yr_offset in range(5):
        yr = start_year + yr_offset
        # En el primer año solo cuentan los meses desde start_month
        if yr == start_year:
            factor = len([m for m in range(1,13)
                          if any(m in DIVIDEND_DATA_FILTERED.get(t,{}).get('dividend_months',[])
                                 for t in tickers) and m >= start_month])
            bruto_yr = total_bruto_first_year
        else:
            bruto_yr = total_bruto

        ret_yr  = bruto_yr * RETENCION_ESPANA
        neto_yr = bruto_yr - ret_yr
        proyeccion_rows.append({
            'Año': str(yr),
            'Bruto': f"€{bruto_yr:,.2f}",
            'Ret. 19%': f"€{ret_yr:,.2f}",
            'Neto': f"€{neto_yr:,.2f}",
            'Media Mensual Neta': f"€{neto_yr/12:,.2f}"
        })

    st.dataframe(pd.DataFrame(proyeccion_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Resumen fiscal ──
    st.markdown("### 🏛️ Resumen Fiscal Anual")
    fiscal_data = []
    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        annual_div = div_data.get('annual_dividend', 0) * cfg['shares']
        ret_esp = annual_div * RETENCION_ESPANA
        fiscal_data.append({
            'ETF': ticker, 'Bruto Anual': f"€{annual_div:.2f}",
            'Ret. 19%': f"€{ret_esp:.2f}", 'Neto Anual': f"€{annual_div - ret_esp:.2f}"
        })
    fiscal_data.append({
        'ETF': '📊 TOTAL', 'Bruto Anual': f"€{total_bruto:.2f}",
        'Ret. 19%': f"€{total_retencion:.2f}", 'Neto Anual': f"€{total_neto:.2f}"
    })
    st.dataframe(pd.DataFrame(fiscal_data), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="background-color:#dcfce7;border:2px solid #16a34a;padding:15px;
                border-radius:10px;margin-top:15px;text-align:center;">
        <span style="color:#166534;font-size:1.1rem;font-weight:600;">
            💰 Ingreso Mensual Medio Neto (año completo):
        </span>
        <span style="color:#166534;font-weight:bold;font-size:1.5rem;margin-left:10px;">
            €{total_neto/12:.2f}
        </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3: IMPORTES EN ESPAÑA
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("💶 Dividendos en España")
    st.info(f"📅 Proyección desde: **{start_date.strftime('%d/%m/%Y')}**")

    st.markdown("""
    <div style="background-color:#fef3c7;border-left:4px solid #f59e0b;padding:12px;
                border-radius:0 8px 8px 0;margin-bottom:15px;">
        <b>🇪🇸 Fiscalidad:</b> ETFs UCITS sin retención origen (0%).
        España retiene <b>19%</b>. Neto = Bruto × 0.81
    </div>""", unsafe_allow_html=True)

    total_bruto_esp = 0
    total_first_year_esp = 0
    spain_data = []

    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        annual_bruto    = div_data.get('annual_dividend', 0) * cfg['shares']
        first_yr_bruto  = div_data.get('first_year_dividend', 0) * cfg['shares']
        total_bruto_esp    += annual_bruto
        total_first_year_esp += first_yr_bruto
        ret_espana  = annual_bruto * RETENCION_ESPANA
        neto_espana = annual_bruto - ret_espana
        first_yr_neto = first_yr_bruto * (1 - RETENCION_ESPANA)
        spain_data.append({
            'ETF': ticker,
            'Nombre': cfg['name'][:20],
            'Domicilio': 'Irlanda' if cfg['isin'].startswith('IE') else 'P. Bajos',
            'Bruto Anual': f"€{annual_bruto:.2f}",
            'Ret. 19%': f"€{ret_espana:.2f}",
            'Neto Anual': f"€{neto_espana:.2f}",
            '1er Año Neto': f"€{first_yr_neto:.2f}",
        })

    total_ret_esp       = total_bruto_esp * RETENCION_ESPANA
    total_neto_esp      = total_bruto_esp - total_ret_esp
    total_first_yr_neto = total_first_year_esp * (1 - RETENCION_ESPANA)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Bruto Anual",    f"€{total_bruto_esp:,.2f}")
    col2.metric("🇪🇸 Ret. 19%",     f"-€{total_ret_esp:,.2f}")
    col3.metric("💰 Neto Anual",     f"€{total_neto_esp:,.2f}")
    col4.metric("🗓️ 1er Año Neto",  f"€{total_first_yr_neto:,.2f}")

    st.markdown("---")
    st.markdown("### 📋 Desglose por ETF")
    st.dataframe(pd.DataFrame(spain_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"### 📅 Cobros Mensuales — Ciclo Anual Completo")
    st.caption("Se muestran todos los meses de pago. Verde = futuro, Amarillo = ya pasó este año.")

    # Reconstruir monthly_spain con TODOS los meses (sin filtro de start_month)
    monthly_spain = {i: {'bruto': 0.0, 'neto': 0.0, 'etfs': []} for i in range(1, 13)}

    for ticker, cfg in PORTFOLIO_CONFIG.items():
        div_data = DIVIDEND_DATA_FILTERED.get(ticker, {})
        div_per_payment = div_data.get('div_per_payment', 0)
        div_months      = div_data.get('dividend_months', [])
        annual_div      = div_data.get('annual_dividend', 0)
        payments_per_year = len(div_months) if div_months else 1
        if div_per_payment == 0:
            div_per_payment = annual_div / payments_per_year if payments_per_year > 0 else 0
        payment_total = div_per_payment * cfg['shares']

        for month in div_months:
            monthly_spain[month]['bruto'] += payment_total
            monthly_spain[month]['neto']  += payment_total * (1 - RETENCION_ESPANA)
            monthly_spain[month]['etfs'].append(ticker.split('.')[0])

    cols = st.columns(4)
    for i in range(12):
        mes_num = i + 1
        with cols[i % 4]:
            data = monthly_spain[mes_num]
            es_futuro = mes_num >= today_month

            if data['bruto'] > 0 and es_futuro:
                bg, border = "#dcfce7", "#16a34a"
            elif data['bruto'] > 0:
                bg, border = "#fef3c7", "#f59e0b"
            else:
                bg, border = "#f1f5f9", "#94a3b8"

            etfs_str = ", ".join(data['etfs']) if data['etfs'] else "—"
            st.markdown(f"""
            <div style="background-color:{bg};border:2px solid {border};border-radius:8px;
                        padding:12px;margin:3px 0;text-align:center;min-height:130px;">
                <div style="font-weight:bold;color:#1e293b;font-size:0.9rem;margin-bottom:4px;">
                    {MESES[i]}</div>
                <div style="color:#64748b;font-size:0.72rem;">Bruto: €{data['bruto']:.2f}</div>
                <div style="color:#dc2626;font-size:0.72rem;">
                    -€{data['bruto']*RETENCION_ESPANA:.2f}</div>
                <div style="color:#16a34a;font-weight:700;font-size:1.1rem;margin-top:3px;">
                    €{data['neto']:.2f}</div>
                <div style="color:#64748b;font-size:0.65rem;margin-top:3px;">{etfs_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    avg_mensual_neto = total_neto_esp / 12
    yield_neto = (total_neto_esp / CAPITAL_INVERTIDO) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color:#fff;border:2px solid #2563eb;padding:15px;border-radius:10px;">
            <h4 style="color:#1e293b;margin-bottom:10px;">📊 Totales Anuales</h4>
            <table style="width:100%;color:#334155;font-size:0.9rem;">
                <tr><td>Dividendos Brutos:</td>
                    <td style="text-align:right;font-weight:bold;">€{total_bruto_esp:,.2f}</td></tr>
                <tr><td>Retención 19%:</td>
                    <td style="text-align:right;color:#dc2626;">-€{total_ret_esp:,.2f}</td></tr>
                <tr style="border-top:1px solid #e2e8f0;">
                    <td style="padding-top:8px;"><b>NETO:</b></td>
                    <td style="text-align:right;font-weight:bold;color:#16a34a;padding-top:8px;">
                        €{total_neto_esp:,.2f}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:#fff;border:2px solid #16a34a;padding:15px;border-radius:10px;">
            <h4 style="color:#1e293b;margin-bottom:10px;">📅 Promedios</h4>
            <table style="width:100%;color:#334155;font-size:0.9rem;">
                <tr><td>Media Mensual Neto:</td>
                    <td style="text-align:right;font-weight:bold;color:#16a34a;">
                        €{avg_mensual_neto:,.2f}</td></tr>
                <tr><td>Yield Bruto:</td>
                    <td style="text-align:right;">
                        {(total_bruto_esp/CAPITAL_INVERTIDO)*100:.2f}%</td></tr>
                <tr><td>Yield Neto:</td>
                    <td style="text-align:right;font-weight:bold;">{yield_neto:.2f}%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#eff6ff;border-left:4px solid #3b82f6;padding:12px;
                border-radius:0 8px 8px 0;margin-top:15px;font-size:0.85rem;">
        <b>📝 Notas:</b><br>
        • Datos de dividendos obtenidos de digrin.com (con fallback histórico)<br>
        • Importes basados en dividendos históricos recientes (pueden variar)<br>
        • El 19% retenido es a cuenta del IRPF (se regulariza en declaración)<br>
        • ETFs domiciliados en P. Bajos (TDIV) pueden tener retención 15% recuperable<br>
        • El gráfico de rendimiento incluye dividendos cobrados (Total Return)
    </div>""", unsafe_allow_html=True)
