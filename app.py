import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS ---
st.set_page_config(page_title="Portfolio Master", layout="wide", page_icon="💎")

# CSS para "tunear" la interfaz (Estilo Dashboard Profesional)
st.markdown("""
<style>
    /* Fuente y fondos */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Estilo para las Tarjetas de Métricas (KPIs) */
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-delta-pos {
        color: #34d399; /* Verde menta */
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #f87171; /* Rojo suave */
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Títulos */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- DATOS DE LA CARTERA ---
PORTFOLIO_CONFIG = {
    'DJMC.AS': {'name': 'iShares EURO STOXX Mid', 'weight': 0.30, 'color': '#3b82f6'},
    'EXH9.DE': {'name': 'iShares STOXX Eu 600 Util', 'weight': 0.129, 'color': '#10b981'},
    'ISPA.DE': {'name': 'iShares Global Sel Div 100', 'weight': 0.171, 'color': '#f59e0b'},
    'IUSM.DE': {'name': 'iShares USD Treasury 7-10yr', 'weight': 0.30, 'color': '#8b5cf6'},
    'SYBJ.DE': {'name': 'SPDR Euro High Yield Bond', 'weight': 0.10, 'color': '#ec4899'}
}

RISK_FREE_RATE = 0.03

# --- FUNCIONES ---
def get_data_safe(tickers, start_date):
    try:
        # auto_adjust=True soluciona el problema del 'Adj Close' y el KeyError
        data = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)
        if 'Close' in data.columns:
            return data['Close']
        else:
            # Fallback por si la estructura cambia
            return data
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def calculate_kpis(series):
    if series.empty: return 0, 0, 0, pd.Series()
    
    # Retornos
    returns = series.pct_change().dropna()
    
    # CAGR
    days = (series.index[-1] - series.index[0]).days
    years = days / 365.25
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max Drawdown
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Sharpe
    excess_returns = returns - (RISK_FREE_RATE / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
    
    return cagr, max_dd, sharpe, drawdown

# --- UI: SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("---")
    initial_capital = st.number_input("Capital Inicial (€)", value=13000, step=1000)
    start_date = st.date_input("Fecha de Inicio", value=datetime.now() - timedelta(days=365*2))
    
    st.markdown("### 🎯 Objetivo")
    df_alloc = pd.DataFrame.from_dict(PORTFOLIO_CONFIG, orient='index')
    df_alloc['Objetivo'] = df_alloc['weight'].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_alloc[['name', 'Objetivo']], use_container_width=True, hide_index=False)
    
    if st.button("🔄 Actualizar Datos", type="primary"):
        st.rerun()

# --- LÓGICA PRINCIPAL ---
st.title("🚀 Dashboard de Control de Cartera")
st.markdown(f"Estrategia: **Bandas ±10%** | Capital Asignado: **{initial_capital:,.0f} €**")

# Descarga de datos
tickers = list(PORTFOLIO_CONFIG.keys())
df_prices = get_data_safe(tickers, start_date)

if not df_prices.empty:
    # Rellenar datos faltantes (forward fill) para evitar huecos en festivos distintos
    df_prices = df_prices.fillna(method='ffill').dropna()

    # Simulación de Compra (Backtest rápido desde fecha inicio)
    initial_prices = df_prices.iloc[0]
    current_prices = df_prices.iloc[-1]
    
    shares = {}
    current_vals = {}
    portfolio_series = pd.DataFrame()
    
    total_current_value = 0
    
    # Calcular valor histórico de cada posición
    for ticker, info in PORTFOLIO_CONFIG.items():
        # Acciones compradas teóricamente el día 1
        num_shares = (initial_capital * info['weight']) / initial_prices[ticker]
        shares[ticker] = num_shares
        
        # Serie histórica de valor
        portfolio_series[ticker] = df_prices[ticker] * num_shares
        
        # Valor actual
        val = num_shares * current_prices[ticker]
        current_vals[ticker] = val
        total_current_value += val

    portfolio_series['Total'] = portfolio_series.sum(axis=1)
    
    # Métricas
    cagr, max_dd, sharpe, dd_series = calculate_kpis(portfolio_series['Total'])
    pnl_abs = total_current_value - initial_capital
    pnl_rel = (total_current_value / initial_capital) - 1

    # --- SECCIÓN 1: TARJETAS VISUALES (HTML/CSS INYECTADO) ---
    st.markdown("### 📊 Métricas Clave")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    def create_card(label, value, subtext, is_positive=True):
        color_class = "metric-delta-pos" if is_positive else "metric-delta-neg"
        html = f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{color_class}">{subtext}</div>
        </div>
        """
        return html

    with kpi1:
        st.markdown(create_card("Valor Cartera", f"{total_current_value:,.0f} €", f"{pnl_abs:+,.0f} € ({pnl_rel:+.2%})", pnl_abs >= 0), unsafe_allow_html=True)
    with kpi2:
        st.markdown(create_card("CAGR (Anual)", f"{cagr:.2%}", "Crecimiento Compuesto", cagr > 0), unsafe_allow_html=True)
    with kpi3:
        st.markdown(create_card("Max Drawdown", f"{max_dd:.2%}", "Caída Máxima Histórica", False), unsafe_allow_html=True)
    with kpi4:
        st.markdown(create_card("Ratio Sharpe", f"{sharpe:.2f}", "Rentabilidad / Riesgo", sharpe > 1), unsafe_allow_html=True)

    st.markdown("---")

    # --- SECCIÓN 2: GRÁFICOS ---
    col_chart, col_rebal = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📈 Evolución Patrimonial")
        
        # Gráfico de Área con degradado (simulado con Plotly)
        fig = px.area(portfolio_series, x=portfolio_series.index, y='Total')
        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="Valor (€)",
            hovermode="x unified",
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.2)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown pequeño abajo
        with st.expander("📉 Ver Gráfico de Riesgo (Drawdown)"):
            fig_dd = px.area(dd_series, title="")
            fig_dd.update_traces(line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.2)')
            fig_dd.add_hrect(y0=-1, y1=-0.20, fillcolor="red", opacity=0.1, line_width=0)
            fig_dd.update_layout(yaxis_title="Caída desde Máx.", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dd, use_container_width=True)

    # --- SECCIÓN 3: REBALANCEO (El Cerebro) ---
    with col_rebal:
        st.subheader("⚖️ Semáforo de Rebalanceo")
        st.caption("Bandas de tolerancia: ±10% relativo al peso objetivo.")
        
        rebal_list = []
        
        for ticker, info in PORTFOLIO_CONFIG.items():
            current_w = current_vals[ticker] / total_current_value
            target_w = info['weight']
            
            # Cálculos de banda
            upper = target_w * 1.10
            lower = target_w * 0.90
            
            action_label = "MANTENER"
            action_color = "#10b981" # Green
            diff_euro = 0
            
            if current_w > upper:
                action_label = "VENDER"
                action_color = "#ef4444" # Red
                diff_euro = current_vals[ticker] - (total_current_value * target_w)
                msg = f"Vender {diff_euro:,.0f}€"
            elif current_w < lower:
                action_label = "COMPRAR"
                action_color = "#3b82f6" # Blue
                diff_euro = (total_current_value * target_w) - current_vals[ticker]
                msg = f"Comprar {diff_euro:,.0f}€"
            else:
                msg = "En rango"

            rebal_list.append({
                "Activo": ticker,
                "Peso Real": current_w,
                "Target": target_w,
                "Acción": action_label,
                "Detalle": msg
            })
            
        # Crear Dataframe visual
        df_rb = pd.DataFrame(rebal_list)
        
        # Estilo personalizado para la tabla usando Pandas Styler
        def style_actions(val):
            color = '#10b981' if val == 'MANTENER' else '#ef4444' if val == 'VENDER' else '#3b82f6'
            return f'color: white; background-color: {color}; font-weight: bold; border-radius: 4px; padding: 2px 5px'

        st.dataframe(
            df_rb.style
            .format({"Peso Real": "{:.1%}", "Target": "{:.1%}"})
            .applymap(style_actions, subset=['Acción']),
            use_container_width=True,
            height=350,
            column_config={
                "Activo": st.column_config.TextColumn("Ticker", width="small"),
                "Acción": st.column_config.Column("Estado", width="small"),
            }
        )

        # Gráfico de Donut pequeño
        fig_donut = px.pie(
            values=[current_vals[t] for t in tickers], 
            names=[PORTFOLIO_CONFIG[t]['name'] for t in tickers],
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=0, l=0, r=0), height=200, 
                                paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_donut, use_container_width=True)

else:
    st.warning("⚠️ No se pudieron descargar datos. Revisa tu conexión o los Tickers.")
