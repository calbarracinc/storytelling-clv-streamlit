
import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------
# Configuración general
# ---------------------------------------
st.set_page_config(page_title="Storytelling CLV", page_icon="📈", layout="wide")

st.title("📈 Storytelling de Customer Lifetime Value (CLV)")
st.caption("Interactivo · 3+ gráficos · Insights accionables")

with st.expander("ℹ️ Instrucciones rápidas", expanded=False):
    st.markdown("""
    - El app intenta leer **`CLV.csv`** desde la raíz del repositorio.
    - Si no encuentra el archivo, puedes **subirlo** en la barra lateral.
    - En la barra lateral selecciona las columnas clave (ID, CLV, Fecha, etc.) si no se detectan automáticamente.
    """)

# ---------------------------------------
# Utilidades
# ---------------------------------------
def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def guess_col(candidates, columns):
    cols_norm = {c: _norm(c) for c in columns}
    for cand in candidates:
        candn = _norm(cand)
        # 1) match exact normalizado
        for c, cn in cols_norm.items():
            if cn == candn:
                return c
        # 2) match por inclusión
        for c, cn in cols_norm.items():
            if candn in cn or cn in candn:
                return c
    return None

def compute_gini(x):
    # Gini robusto: x >= 0
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else np.nan
    return gini

# ---------------------------------------
# Carga de datos
# ---------------------------------------
st.sidebar.header("📁 Datos")
path_default = "CLV.csv"
df = None

if os.path.exists(path_default):
    try:
        df = pd.read_csv(path_default)
        st.sidebar.success("Leído archivo local: CLV.csv")
    except Exception as e:
        st.sidebar.error(f"No se pudo leer CLV.csv: {e}")

if df is None:
    up = st.sidebar.file_uploader("Sube tu CLV.csv", type=["csv"])
    if up:
        df = pd.read_csv(up)

if df is None:
    st.warning("Sube un archivo **CLV.csv** o colócalo en la raíz del repositorio para continuar.")
    st.stop()

st.write("**Vista previa de datos (primeras filas):**")
st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------
# Selección de columnas
# ---------------------------------------
st.sidebar.header("🔧 Selección de columnas")

# Candidatos por tipo
cands_id = ["customer_id","cliente","id_cliente","id","customer","user_id","account_id"]
cands_clv = ["clv","lifetimevalue","customerlifetimevalue","valorvida","valorcliente","customervalue","ingresostotales","revenuetotal","monto","monto_total","sales_total"]
cands_date = ["date","fecha","orderdate","purchasedate","transdate","month","period","fechacompra"]
cands_freq = ["frequency","orders","ordercount","compras","num_pedidos","n_ordenes","transacciones","visits","frecuencia"]
cands_rev = ["revenue","ingresos","ventas","sales","total_spent","amount","facturacion","monto","ingreso"]

# Intentar adivinar
id_guess = guess_col(cands_id, df.columns)
clv_guess = guess_col(cands_clv, df.columns)
date_guess = guess_col(cands_date, df.columns)
freq_guess = guess_col(cands_freq, df.columns)
rev_guess = guess_col(cands_rev, df.columns)

col_id = st.sidebar.selectbox("ID de cliente", options=["(ninguna)"] + list(df.columns), index=(["(ninguna)"] + list(df.columns)).index(id_guess) if id_guess in df.columns else 0)
col_clv = st.sidebar.selectbox("Columna CLV (numérica)", options=["(ninguna)"] + list(df.columns), index=(["(ninguna)"] + list(df.columns)).index(clv_guess) if clv_guess in df.columns else 0)
col_date = st.sidebar.selectbox("Fecha (opcional)", options=["(ninguna)"] + list(df.columns), index=(["(ninguna)"] + list(df.columns)).index(date_guess) if date_guess in df.columns else 0)
col_freq = st.sidebar.selectbox("Frecuencia / # compras (opcional)", options=["(ninguna)"] + list(df.columns), index=(["(ninguna)"] + list(df.columns)).index(freq_guess) if freq_guess in df.columns else 0)
col_rev = st.sidebar.selectbox("Ingresos / Monto (opcional)", options=["(ninguna)"] + list(df.columns), index=(["(ninguna)"] + list(df.columns)).index(rev_guess) if rev_guess in df.columns else 0)

# Casts / preprocesamiento
if col_clv != "(ninguna)":
    df[col_clv] = pd.to_numeric(df[col_clv], errors="coerce")

if col_date != "(ninguna)":
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

# Nivel cliente para analítica
if col_id != "(ninguna)" and col_clv != "(ninguna)":
    # Asumimos CLV a nivel cliente -> tomar el valor máximo o último por cliente si hay duplicados
    df_cust = df[[col_id, col_clv]].dropna().groupby(col_id, as_index=False)[col_clv].max()
else:
    # Intentar derivar un CLV proxy si no hay CLV explícito
    if col_id != "(ninguna)" and col_rev != "(ninguna)":
        df_cust = df[[col_id, col_rev]].dropna().groupby(col_id, as_index=False)[col_rev].sum().rename(columns={col_rev: "CLV_proxy"})
        col_clv = "CLV_proxy"
        st.info("No se proporcionó CLV. Se usará **suma de ingresos** como CLV proxy a nivel cliente.")
    else:
        st.error("Selecciona al menos **ID de cliente** y **CLV** (o **Ingresos** para CLV proxy).")
        st.stop()

# Frecuencia por cliente (si existe)
if col_freq != "(ninguna)" and col_id != "(ninguna)":
    df_freq = df[[col_id, col_freq]].dropna().groupby(col_id, as_index=False)[col_freq].max()
    df_cust = df_cust.merge(df_freq, on=col_id, how="left")
else:
    # derivar frecuencia como número de filas por cliente si posible
    if col_id != "(ninguna)":
        freq_der = df.groupby(col_id).size().reset_index(name="frequency_derived")
        df_cust = df_cust.merge(freq_der, on=col_id, how="left")
        if col_freq == "(ninguna)":
            col_freq = "frequency_derived"

# ---------------------------------------
# KPIs e Insights
# ---------------------------------------
kpi_cols = st.columns(5)
n_clients = df_cust[col_id].nunique()
mean_clv = float(np.nanmean(df_cust[col_clv]))
median_clv = float(np.nanmedian(df_cust[col_clv]))
p90 = float(np.nanpercentile(df_cust[col_clv], 90))
gini = float(compute_gini(df_cust[col_clv]))

with kpi_cols[0]:
    st.metric("Clientes únicos", f"{n_clients:,}")
with kpi_cols[1]:
    st.metric("CLV promedio", f"{mean_clv:,.2f}")
with kpi_cols[2]:
    st.metric("CLV mediano", f"{median_clv:,.2f}")
with kpi_cols[3]:
    st.metric("P90 CLV", f"{p90:,.2f}")
with kpi_cols[4]:
    st.metric("Índice Gini (concentración)", f"{gini:0.2f}" if not np.isnan(gini) else "NA")

# Pareto: ordenar clientes por CLV
df_pareto = df_cust.sort_values(by=col_clv, ascending=False).reset_index(drop=True)
df_pareto["cum_clv"] = df_pareto[col_clv].cumsum()
total_clv = df_pareto[col_clv].sum()
df_pareto["cum_share"] = df_pareto["cum_clv"] / total_clv if total_clv > 0 else 0
df_pareto["cust_share"] = (df_pareto.index + 1) / len(df_pareto)

# % de clientes necesario para alcanzar 80% del CLV
pct_clients_80 = float((df_pareto["cum_share"] >= 0.80).idxmin() + 1) / len(df_pareto) * 100 if total_clv > 0 else np.nan

st.markdown(f"""
**Insight clave:** El **{pct_clients_80:0.1f}%** de clientes concentra ~**80%** del CLV total 
(ley de Pareto). Esto orienta estrategias de **retención** y **upsell** focalizadas.
""")

# ---------------------------------------
# "Slides" del storytelling
# ---------------------------------------
st.sidebar.header("🖼️ Navegación de Storytelling")
slide = st.sidebar.radio("Ir a:", options=[
    "Slide 1 · Distribución CLV",
    "Slide 2 · Pareto de clientes",
    "Slide 3 · Evolución temporal",
    "Slide 4 · CLV vs. Frecuencia (opcional)",
])

# ---------- Slide 1: Distribución
if slide == "Slide 1 · Distribución CLV":
    st.subheader("Slide 1 · Distribución del CLV")
    nbins = st.slider("Número de bins (histograma)", 10, 80, 40)
    fig1 = px.histogram(df_cust, x=col_clv, nbins=nbins, marginal="box", opacity=0.85)
    fig1.update_layout(yaxis_title="Clientes", xaxis_title="CLV", bargap=0.05)
    st.plotly_chart(fig1, use_container_width=True)

    # Narrativa
    st.markdown("""
    **¿Qué mirar?**
    - Identifica **outliers** (clientes extremadamente valiosos).
    - Compara **promedio vs. mediana** para ver sesgo por pocos clientes de alto CLV.
    - Ajusta bins para ver la **dispersión**.
    """)

# ---------- Slide 2: Pareto
elif slide == "Slide 2 · Pareto de clientes":
    st.subheader("Slide 2 · Curva de Pareto (CLV acumulado)")
    top_n = st.slider("Mostrar Top N clientes en barras (para detalle)", 10, min(300, len(df_pareto)), min(50, len(df_pareto)))
    show_line_full = st.checkbox("Mostrar línea de acumulado en toda la base", True)

    bars = go.Bar(x=list(range(1, top_n+1)), y=df_pareto[col_clv].head(top_n), name="CLV por cliente")
    line = go.Scatter(x=np.arange(1, len(df_pareto)+1) if show_line_full else np.arange(1, top_n+1),
                      y=(df_pareto["cum_share"]*100) if show_line_full else (df_pareto["cum_share"].head(top_n)*100),
                      name="CLV acumulado (%)", mode="lines+markers", yaxis="y2")
    layout = go.Layout(
        xaxis=dict(title="Clientes (ordenados por CLV)"),
        yaxis=dict(title="CLV"),
        yaxis2=dict(title="CLV acumulado (%)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h"),
    )
    fig2 = go.Figure(data=[bars, line], layout=layout)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    **Lectura:** Concentra esfuerzos en el segmento superior; aproximadamente el **{pct_clients_80:0.1f}%** de clientes produce el **80%** del valor.
    - Define **beneficios exclusivos**, programas **VIP** y **campañas 1:1** para ese grupo.
    - Para la cola larga, considera **automatización**, **cross-sell** simple y **promos** de baja fricción.
    """)

# ---------- Slide 3: Evolución temporal
elif slide == "Slide 3 · Evolución temporal":
    st.subheader("Slide 3 · Evolución temporal")

    if col_date == "(ninguna)":
        st.info("Selecciona una **columna de Fecha** en la barra lateral para habilitar esta slide.")
    else:
        # Si hay ingresos, usar ingresos mensuales; si no, usar promedio de CLV mensual
        df_time = df.copy()
        df_time = df_time.dropna(subset=[col_date])
        df_time["month"] = pd.to_datetime(df_time[col_date]).dt.to_period("M").dt.to_timestamp()

        if col_rev != "(ninguna)":
            df_time[col_rev] = pd.to_numeric(df_time[col_rev], errors="coerce")
            serie = df_time.groupby("month")[col_rev].sum().reset_index()
            y_col = col_rev
            y_title = "Ingresos mensuales"
        else:
            serie = df_time.groupby("month")[col_clv].mean().reset_index()
            y_col = col_clv
            y_title = "CLV promedio mensual"

        fig3 = px.line(serie, x="month", y=y_col, markers=True)
        fig3.update_layout(xaxis_title="Mes", yaxis_title=y_title)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        **¿Qué mirar?**
        - Estacionalidad, picos y caídas.
        - Impacto de campañas, lanzamientos o cambios de precio.
        - Si tienes ingresos, monitorea **tendencia de facturación**; si no, el **CLV promedio** mensual.
        """)

# ---------- Slide 4: Scatter CLV vs frecuencia
elif slide == "Slide 4 · CLV vs. Frecuencia (opcional)":
    st.subheader("Slide 4 · Relación CLV vs. Frecuencia")
    if col_freq == "(ninguna)":
        st.info("Selecciona o deja que el app derive una **Frecuencia** en la barra lateral para habilitar esta slide.")
    else:
        df_sc = df_cust[[col_id, col_clv, col_freq]].dropna()
        fig4 = px.scatter(df_sc, x=col_freq, y=col_clv, hover_data=[col_id], trendline=None)
        fig4.update_layout(xaxis_title="Frecuencia (# compras)", yaxis_title="CLV")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        **Insight común:** clientes con mayor **frecuencia** suelen tener **CLV** más alto.  
        - Crea **hábitos de compra** (suscripciones, packs).
        - Remueve fricciones en **checkout** y **entregas**.
        """)

# ---------------------------------------
# Descargas
# ---------------------------------------
st.sidebar.header("⬇️ Descargas")
# Diccionario de KPIs para CSV
insights = {
    "clientes_unicos": n_clients,
    "clv_promedio": round(mean_clv, 2),
    "clv_mediano": round(median_clv, 2),
    "p90_clv": round(p90, 2),
    "gini": round(gini, 4) if not np.isnan(gini) else "",
    "pct_clientes_para_80pct_clv": round(pct_clients_80, 2) if not np.isnan(pct_clients_80) else ""
}
insights_csv = pd.DataFrame([insights]).to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Descargar KPIs (CSV)", data=insights_csv, file_name="insights_clv.csv", mime="text/csv")

# También permitir exportar df_cust con CLV y frecuencia
cust_csv = df_cust.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Descargar dataset cliente (CSV)", data=cust_csv, file_name="clientes_clv.csv", mime="text/csv")

st.success("✅ Storytelling listo. Navega los **Slides** desde la barra lateral.")
