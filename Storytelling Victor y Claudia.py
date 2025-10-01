
# -*- coding: utf-8 -*-
"""
Storytelling CLV (4 slides, sin cargador de archivos)
Lee CLV.csv desde la raíz del repositorio (no pide upload).
Autoría original: Victor Garcia & Claudia Albarracín
Ajustes: app con 4 visualizaciones + narrativa
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(page_title="Storytelling CLV", page_icon="📈", layout="wide")
st.title("📈 Storytelling de Customer Lifetime Value (CLV)")
st.caption("4 slides · insights accionables · no requiere subir archivo (usa CLV.csv del repo)")

# -----------------------------
# Utilidades
# -----------------------------
def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def guess_col(candidates, columns):
    cols_norm = {c: _norm(c) for c in columns}
    for cand in candidates:
        candn = _norm(cand)
        for c, cn in cols_norm.items():
            if cn == candn:
                return c
        for c, cn in cols_norm.items():
            if candn in cn or cn in candn:
                return c
    return None

def to_numeric_robust(series):
    """Convierte a numérico manejando puntos/comas y símbolos."""
    s = series.astype(str).str.replace(r"[^0-9,.\-]", "", regex=True)
    # si hay comas y puntos, asumir '.' miles y ',' decimal
    mask_comma_decimal = s.str.contains(",") & s.str.contains(r"\.")
    s = np.where(mask_comma_decimal, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), s)
    # si solo hay comas, asume coma decimal
    only_comma = s.str.contains(",") & ~s.str.contains(r"\.")
    s = np.where(only_comma, s.str.replace(",", ".", regex=False), s)
    # remover separadores de miles restantes
    s = pd.Series(s).str.replace(r"(?<=\d)[ ](?=\d{3}\b)", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def compute_gini(x):
    x = np.array(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), dtype=float)
    if x.size == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else np.nan

# -----------------------------
# Carga del dataset local
# -----------------------------
DATA_PATHS = ["CLV.csv", "./data/CLV.csv", "./dataset/CLV.csv"]
df = None
err = None

for p in DATA_PATHS:
    if os.path.exists(p):
        try:
            # Intento 1: ;
            df = pd.read_csv(p, sep=";")
        except Exception:
            try:
                # Intento 2: ,
                df = pd.read_csv(p)
            except Exception as e:
                err = f"No pude leer {p}: {e}"
                df = None
        if df is not None:
            break

if df is None:
    st.error("No se encontró **CLV.csv** en la raíz del repo. Súbelo y vuelve a ejecutar.")
    if err:
        st.caption(err)
    st.stop()

# limpieza básica de columnas
df.columns = [str(c).strip() for c in df.columns]

# Intentar adivinar campos típicos
cands_clv   = ["customer lifetime value","clv","valorvida","valorcliente","ingresos totales","revenue","amount"]
cands_date  = ["effective to date","fecha","date","order date","purchase date"]
cands_state = ["state","estado","region"]
cands_resp  = ["response","respuesta"]
cands_income= ["income","ingreso"]
cands_vclass= ["vehicle class","vehicleclass","clase vehiculo"]

col_clv    = guess_col(cands_clv, df.columns) or "Customer Lifetime Value"
col_date   = guess_col(cands_date, df.columns) or "Effective To Date"
col_state  = guess_col(cands_state, df.columns) or "State"
col_resp   = guess_col(cands_resp, df.columns) or "Response"
col_income = guess_col(cands_income, df.columns) or "Income"
col_vclass = guess_col(cands_vclass, df.columns) or "Vehicle Class"

# Casts robustos
if col_clv in df.columns:
    if not np.issubdtype(df[col_clv].dtype, np.number):
        df[col_clv] = to_numeric_robust(df[col_clv])
else:
    st.error("No encontré la columna de **CLV**. Renómbrala o ajusta el código.")
    st.stop()

if col_date in df.columns:
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

# KPIs
n_rows = len(df)
mean_clv = float(np.nanmean(df[col_clv]))
median_clv = float(np.nanmedian(df[col_clv]))
p90 = float(np.nanpercentile(df[col_clv].dropna(), 90)) if df[col_clv].notna().any() else np.nan
gini = float(compute_gini(df[col_clv]))

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Registros", f"{n_rows:,}")
k2.metric("CLV promedio", f"{mean_clv:,.2f}")
k3.metric("CLV mediano", f"{median_clv:,.2f}")
k4.metric("P90 CLV", f"{p90:,.2f}")
k5.metric("Índice Gini", f"{gini:0.2f}" if not np.isnan(gini) else "NA")

st.divider()
st.write("**Vista previa de datos:**")
st.dataframe(df.head(15), use_container_width=True)

# -----------------------------
# Slides (exactamente 4)
# -----------------------------
st.sidebar.header("🖼️ Navegación")
slide = st.sidebar.radio("Ir a:", [
    "Slide 1 · Distribución del CLV",
    "Slide 2 · Pareto (acumulado CLV)",
    "Slide 3 · Evolución temporal",
    "Slide 4 · CLV vs Ingreso / Segmentos",
])

# ---------- Slide 1
if slide == "Slide 1 · Distribución del CLV":
    st.subheader("Slide 1 · Distribución del CLV (histograma + box)")
    nbins = st.slider("Número de bins", 10, 80, 40)
    fig = px.histogram(df, x=col_clv, nbins=nbins, marginal="box", opacity=0.85)
    fig.update_layout(yaxis_title="Observaciones", xaxis_title="CLV", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**¿Qué mirar?**
- Outliers (clientes/pólizas extremadamente valiosas).
- Sesgo: compara **promedio vs. mediana** (si hay cola larga a la derecha).
- Usa los *bins* para afinar la lectura por rangos de valor.
""")

# ---------- Slide 2
elif slide == "Slide 2 · Pareto (acumulado CLV)":
    st.subheader("Slide 2 · Curva de Pareto (acumulado del CLV)")
    df_p = df[[col_clv]].dropna().sort_values(by=col_clv, ascending=False).reset_index(drop=True)
    if df_p.empty:
        st.info("No hay valores válidos de CLV para construir Pareto.")
    else:
        df_p["cum_clv"] = df_p[col_clv].cumsum()
        total = df_p[col_clv].sum()
        df_p["cum_share"] = df_p["cum_clv"] / total if total>0 else 0
        df_p["rank"] = np.arange(1, len(df_p)+1)
        # % necesario para 80%
        idx80 = int(np.searchsorted(df_p["cum_share"].values, 0.80, side="left"))
        idx80 = min(idx80, len(df_p)-1)
        pct_clients_80 = (idx80+1)/len(df_p)*100

        max_top = max(1, min(300, len(df_p)))
        top_n = st.slider("Top N para barras (detalle)", 1, max_top, min(50, max_top))

        bars = go.Bar(x=df_p["rank"].head(top_n), y=df_p[col_clv].head(top_n), name="CLV por registro")
        line = go.Scatter(x=df_p["rank"], y=df_p["cum_share"]*100, mode="lines", name="Acumulado (%)", yaxis="y2")
        layout = go.Layout(
            xaxis=dict(title="Registros ordenados por CLV"),
            yaxis=dict(title="CLV"),
            yaxis2=dict(title="CLV acumulado (%)", overlaying="y", side="right", range=[0,100]),
            legend=dict(orientation="h")
        )
        fig2 = go.Figure(data=[bars, line], layout=layout)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
**Insight:** Aproximadamente el **{pct_clients_80:0.1f}%** de registros concentra el **80%** del CLV total.
- Enfoca **retención/upsell** en este segmento alto valor.
- Automatiza y simplifica ofertas para la **cola larga**.
""")

# ---------- Slide 3
elif slide == "Slide 3 · Evolución temporal":
    st.subheader("Slide 3 · Evolución temporal del CLV")
    if col_date not in df.columns or df[col_date].isna().all():
        st.info("No hay columna de fecha válida. Verifica que exista y tenga datos.")
    else:
        serie = (
            df.dropna(subset=[col_date, col_clv])
              .groupby(pd.Grouper(key=col_date, freq='M'))[col_clv]
              .sum()
              .reset_index()
        )
        fig3 = px.line(serie, x=col_date, y=col_clv, markers=True)
        fig3.update_layout(xaxis_title="Mes", yaxis_title="CLV mensual")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
**¿Qué mirar?**
- Picos/caídas y su relación con campañas o precios.
- Estacionalidad por meses.
- Si la tendencia cae, prueba **reactivación** y **cross-sell**.
""")

# ---------- Slide 4
elif slide == "Slide 4 · CLV vs Ingreso / Segmentos":
    st.subheader("Slide 4 · Relación CLV vs Ingreso y por segmentos")
    cols = st.columns(2)
    # Dispersión CLV vs Ingreso
    if col_income in df.columns:
        with cols[0]:
            df_sc = df[[col_income, col_clv]].dropna()
            if not df_sc.empty:
                fig4a = px.scatter(df_sc, x=col_income, y=col_clv, trendline=None)
                fig4a.update_layout(xaxis_title="Ingreso", yaxis_title="CLV")
                st.plotly_chart(fig4a, use_container_width=True)
            else:
                st.info("No hay datos suficientes para el scatter CLV vs Ingreso.")
    else:
        with cols[0]:
            st.info("No se encontró columna de Ingreso.")

    # Boxplot por segmento (State o Vehicle Class)
    seg = col_state if col_state in df.columns else (col_vclass if col_vclass in df.columns else None)
    with cols[1]:
        if seg:
            df_b = df[[seg, col_clv]].dropna()
            if not df_b.empty:
                fig4b = px.box(df_b, x=seg, y=col_clv, points=False)
                fig4b.update_layout(xaxis_title=seg, yaxis_title="CLV")
                st.plotly_chart(fig4b, use_container_width=True)
            else:
                st.info("No hay datos suficientes para el boxplot por segmento.")
        else:
            st.info("No se encontró columna de segmento apropiada (State/Vehicle Class).")

    st.markdown("""
**Lectura:**
- La **frecuencia** y el **poder adquisitivo** (Ingreso) suelen correlacionar con CLV.
- Segmentos con mayor mediana CLV merecen **beneficios diferenciados**.
""")
