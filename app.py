import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from pathlib import Path

st.set_page_config(page_title="SaaS Valuation & MRR Dashboard", layout="wide")

# ------------- Helpers -------------
@st.cache_data(show_spinner=False)
def load_excel(file) -> dict:
    xls = pd.ExcelFile(file)
    sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}
    return sheets

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias: {missing} en la hoja.")
        st.stop()

def safe_div(a, b):
    return (a / b) if (b is not None and b != 0 and not pd.isna(b)) else np.nan

def monthly_fifo_cohorts(df):
    """Construye una matriz de cohortes aproximada (FIFO) usando agregados mensuales.
    df requiere columnas: Date (periodic month), Plan, New Customers, Lost Customers
    Devuelve dos pivots: total y por plan (retención %).
    """
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.to_period("M").dt.to_timestamp()
    work = work.sort_values(["Plan", "Date"]).reset_index(drop=True)

    # Estructuras por plan
    results = []  # filas: plan, cohort_month, month, remaining, initial

    for plan, g in work.groupby("Plan", sort=False):
        queue = []  # lista de [cohort_month, remaining]
        # También guardaremos el tamaño inicial por cohorte para calcular %
        initial_map = {}

        for month, gm in g.groupby("Date", sort=True):
            new_c = int(gm["New Customers"].sum())
            lost_c = int(gm["Lost Customers"].sum())

            if new_c > 0:
                queue.append([month, new_c])
                initial_map.setdefault(month, 0)
                initial_map[month] += new_c

            # Aplicamos bajas FIFO
            remaining_to_remove = lost_c
            qi = 0
            while remaining_to_remove > 0 and qi < len(queue):
                cohort_month, remaining = queue[qi]
                take = min(remaining, remaining_to_remove)
                queue[qi][1] -= take
                remaining_to_remove -= take
                if queue[qi][1] == 0:
                    qi += 1
            # Compactar (eliminar cohortes vacías al principio)
            queue = [q for q in queue if q[1] > 0]

            # Registrar estado de cada cohorte en este mes
            for cohort_month, remaining in queue:
                results.append({
                    "Plan": plan,
                    "Cohort": cohort_month,
                    "Month": month,
                    "Remaining": remaining,
                    "Initial": initial_map.get(cohort_month, np.nan)
                })

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    res = pd.DataFrame(results)
    # Calcular edad en meses para pivot
    res["Age (months)"] = ((res["Month"].dt.year - res["Cohort"].dt.year) * 12 +
                           (res["Month"].dt.month - res["Cohort"].dt.month)).astype(int)
    # Filtrar puntos donde Initial esté definido
    res = res[~res["Initial"].isna() & (res["Initial"] > 0)]
    res["Retention %"] = (res["Remaining"] / res["Initial"]) * 100

    # Pivot TOTAL
    total = (res.groupby(["Cohort", "Age (months)"])["Remaining"].sum().reset_index()
               .merge(res.groupby(["Cohort"])["Initial"].sum().reset_index(), on="Cohort", suffixes=("", "_cohort"))
            )
    total["Retention %"] = (total["Remaining"] / total["Initial_cohort"]) * 100
    pivot_total = total.pivot(index="Cohort", columns="Age (months)", values="Retention %").sort_index()

    # Pivot POR PLAN (promedio ponderado por tamaño de cohorte)
    res["Weight"] = res["Initial"]
    res_plan = (res.groupby(["Plan", "Cohort", "Age (months)"])
                  .apply(lambda x: np.average(x["Retention %"], weights=x["Weight"]))
                  .reset_index(name="Retention %"))
    pivot_plan = res_plan.pivot_table(index=["Plan","Cohort"], columns="Age (months)", values="Retention %")

    return pivot_total, pivot_plan

def compute_components(df_data, df_prices):
    d = df_data.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.to_period("M").dt.to_timestamp()
    p = df_prices.copy()
    d = d.merge(p[["Plan", "Price MRR (€)", "Multiple (x ARR)"]], on="Plan", how="left")

    d = d.sort_values(["Plan","Date"]).reset_index(drop=True)

    # Valores base
    d["New MRR (€)"]       = d["New Customers"]  * d["Price MRR (€)"]
    d["Churned MRR (€)"]   = d["Lost Customers"] * d["Price MRR (€)"]
    # MRR real por plan/mes
    if "Real MRR (optional €)" in d.columns and d["Real MRR (optional €)"].notna().any():
        d["MRR (€)"] = d["Real MRR (optional €)"]
    else:
        # si no hay MRR real, aproximamos con base en clientes activos si existe
        if "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
            d["MRR (€)"] = d["Active Customers (optional)"] * d["Price MRR (€)"]
        else:
            # fallback: acumulado
            d["MRR (€)"] = d.groupby("Plan")["New MRR (€)"].cumsum() - d.groupby("Plan")["Churned MRR (€)"].cumsum()

    # Diferencia mensual por plan
    d["ΔMRR (€)"] = d.groupby("Plan")["MRR (€)"].diff().fillna(d["MRR (€)"])

    # Residuo para inferir expansión/contracción
    residual = d["ΔMRR (€)"] - d["New MRR (€)"] + d["Churned MRR (€)"]
    d["Expansion MRR (inferred €)"]  = residual.clip(lower=0)
    d["Downgraded MRR (inferred €)"] = (-residual).clip(lower=0)

    # Clientes activos aprox si no vienen en fichero
    if "Active Customers (optional)" in d.columns and d["Active Customers (optional)"].notna().any():
        d["Active Customers"] = d["Active Customers (optional)"]
    else:
        d["Active Customers"] = d.groupby("Plan").apply(
            lambda g: g["New Customers"].cumsum() - g["Lost Customers"].cumsum()
        ).reset_index(level=0, drop=True)

    return d

def ytd_metrics(monthly, year):
    # monthly ya agregado TOTAL por mes
    this_year = monthly[monthly["Date"].dt.year == year].copy()
    if this_year.empty:
        return {}

    this_year = this_year.sort_values("Date")
    end = this_year.iloc[-1]
    start_mrr = this_year.iloc[0]["Start MRR (€)"] if not pd.isna(this_year.iloc[0]["Start MRR (€)"]) else this_year.iloc[0]["Total MRR (€)"]

    growth_ytd = safe_div(end["Total MRR (€)"] - start_mrr, start_mrr) * 100 if start_mrr else np.nan
    # GRR YTD y NRR YTD aproximando con sumas sobre el año y dividiendo por el MRR de arranque
    churn_plus_contr = this_year["Churned MRR (€)"].sum() + this_year["Downgraded MRR (inferred €)"].sum()
    expansion = this_year["Expansion MRR (inferred €)"].sum()
    grr_ytd = (1 - safe_div(churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan
    nrr_ytd = (1 + safe_div(expansion - churn_plus_contr, start_mrr)) * 100 if start_mrr else np.nan

    net_new_ytd = this_year["Net New MRR (€)"].sum()

    return dict(growth_ytd=growth_ytd, grr_ytd=grr_ytd, nrr_ytd=nrr_ytd, net_new_ytd=net_new_ytd)

# ------------- UI -------------
st.title("📊 SaaS Valuation & MRR Dashboard")
st.caption("Sube tu Excel (hojas mínimas: **Prices** y **Data**). Opcional: **CAC**.")

uploaded = st.file_uploader("Cargar Excel (.xlsx)", type=["xlsx"])

# Ruta relativa al repo para la plantilla de ejemplo
TEMPLATE_PATH = Path(__file__).with_name("SaaS_Final_Template_COMPLETO_with_CAC.xlsx")

example_note = st.expander("¿No tienes el Excel preparado?")
with example_note:
    if TEMPLATE_PATH.exists():
        st.download_button("📥 Descargar plantilla con CAC",
                           data=TEMPLATE_PATH.read_bytes(),
                           file_name="SaaS_Final_Template_COMPLETO_with_CAC.xlsx")
    else:
        st.info("Sube tu propio Excel (hojas Prices y Data). Si quieres, añade una hoja CAC.")

# Cargar hojas
if uploaded is not None:
    sheets = load_excel(uploaded)
else:
    # fallback: cargar plantilla del repo si existe, si no, obligar a subir
    if TEMPLATE_PATH.exists():
        sheets = load_excel(str(TEMPLATE_PATH))
    else:
        st.warning("No se encontró la plantilla en el repo. Sube tu Excel (Prices y Data, opcional CAC).")
        st.stop()

# Validaciones
ensure_columns(sheets["Prices"], ["Plan","Price MRR (€)","Price ARR (€)","Multiple (x ARR)"])
ensure_columns(sheets["Data"], ["Date","Plan","New Customers","Lost Customers","Active Customers (optional)","Real MRR (optional €)"])

# Dataframes base
df_prices = sheets["Prices"].copy()
df_data = sheets["Data"].copy()
df_cac  = sheets.get("CAC", pd.DataFrame(columns=["Date","Sales & Marketing Spend (€)","New Customers (from Data)"])).copy()

# Normalizaciones
df_data["Date"] = pd.to_datetime(df_data["Date"]).dt.to_period("M").dt.to_timestamp()
df_prices["Plan"] = df_prices["Plan"].astype(str)

# Componentes MRR a nivel plan/mes
comp = compute_components(df_data, df_prices)

# Agregados a TOTAL (todas las tarifas)
monthly = (comp.groupby("Date", as_index=False)
            .agg({
                "New Customers":"sum",
                "Lost Customers":"sum",
                "Active Customers":"sum",
                "New MRR (€)":"sum",
                "Expansion MRR (inferred €)":"sum",
                "Churned MRR (€)":"sum",
                "Downgraded MRR (inferred €)":"sum",
                "MRR (€)":"sum"
            })
          )
monthly = monthly.sort_values("Date")
monthly = monthly.rename(columns={"MRR (€)":"Total MRR (€)"})
monthly["Start MRR (€)"] = monthly["Total MRR (€)"].shift(1)
monthly["Net New MRR (€)"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]
                              - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"])
monthly["Total ARR (€)"] = monthly["Total MRR (€)"] * 12
monthly["GRR %"] = (1 - (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"])
                      / monthly["Start MRR (€)"]).replace([np.inf, -np.inf], np.nan) * 100
monthly["NRR %"] = (1 + (monthly["Expansion MRR (inferred €)"] - monthly["Churned MRR (€)"] - monthly["Downgraded MRR (inferred €)"])
                      / monthly["Start MRR (€)"]).replace([np.inf, -np.inf], np.nan) * 100
monthly["MoM Growth %"] = ((monthly["Total MRR (€)"] - monthly["Start MRR (€)"]) / monthly["Start MRR (€)"] * 100).replace([np.inf, -np.inf], np.nan)
monthly["ARPU (€)"] = monthly["Total MRR (€)"] / monthly["Active Customers"].replace(0, np.nan)
monthly["Quick Ratio"] = (monthly["New MRR (€)"] + monthly["Expansion MRR (inferred €)"]) / (monthly["Churned MRR (€)"] + monthly["Downgraded MRR (inferred €)"]).replace(0, np.nan)

# --------- Filtros ---------
years = sorted(monthly["Date"].dt.year.unique())
default_year = years[-1] if years else datetime.now().year
col1, col2, col3 = st.columns([1,1,1])
with col1:
    sel_years = st.multiselect("Año(s)", options=years, default=[default_year] if years else [])
with col2:
    sector = st.selectbox("Sector/Perfil", [
        "Horizontal SaaS", "Vertical SaaS", "PLG", "Enterprise", "Fintech SaaS", "Health SaaS", "DevTools", "Otro"
    ])
with col3:
    gross_margin = st.slider("Margen bruto (%) para LTV", 40, 95, 80, step=1)

base_multiples = {
    "Horizontal SaaS": 10, "Vertical SaaS": 9, "PLG": 12, "Enterprise": 8,
    "Fintech SaaS": 12, "Health SaaS": 9, "DevTools": 10, "Otro": 8
}
base_mult = base_multiples.get(sector, 10)

# Filtrado por años
filt = monthly[monthly["Date"].dt.year.isin(sel_years)].copy() if sel_years else monthly.copy()

# --------- KPIs (top) ---------
last_row = filt.sort_values("Date").iloc[-1] if not filt.empty else monthly.iloc[-1]
active_now = int(last_row["Active Customers"]) if "Active Customers" in last_row else np.nan
mrr_now = float(last_row["Total MRR (€)"]) if "Total MRR (€)" in last_row else np.nan
arr_now = mrr_now * 12 if pd.notna(mrr_now) else np.nan

# CAC desde hoja CAC (en filtros por año)
if not df_cac.empty:
    df_cac["Date"] = pd.to_datetime(df_cac["Date"]).dt.to_period("M").dt.to_timestamp()
    cac_year = df_cac[df_cac["Date"].dt.year.isin(sel_years)] if sel_years else df_cac.copy()
    total_spend = cac_year["Sales & Marketing Spend (€)"].sum(min_count=1)
    if "New Customers" in cac_year.columns:
        total_new = cac_year["New Customers"].sum(min_count=1)
    else:
        total_new = cac_year["New Customers (from Data)"].sum(min_count=1) if "New Customers (from Data)" in cac_year.columns else np.nan
    cac_value = safe_div(total_spend, total_new)
else:
    cac_value = np.nan

# Churn clientes medio del periodo filtrado
avg_active = filt["Active Customers"].replace(0,np.nan).mean()
churn_rate = safe_div(filt["Lost Customers"].sum(), avg_active)  # mensual aprox
arpu_now = safe_div(mrr_now, active_now)
ltv_value = safe_div(arpu_now * (gross_margin/100), churn_rate)
ltv_cac_ratio = safe_div(ltv_value, cac_value)

# YTD (cogemos el último año seleccionado si hay varios)
ytd = ytd_metrics(monthly, sel_years[-1] if sel_years else default_year)

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Clientes activos", f"{active_now:.0f}" if pd.notna(active_now) else "—")
k2.metric("MRR", f"€ {mrr_now:,.0f}".replace(",", ".")) if pd.notna(mrr_now) else k2.metric("MRR", "—")
k3.metric("ARR", f"€ {arr_now:,.0f}".replace(",", ".")) if pd.notna(arr_now) else k3.metric("ARR","—")
k4.metric("LTV/CAC", f"{ltv_cac_ratio:.2f}" if pd.notna(ltv_cac_ratio) else "—")
k5.metric("LTV (medio)", f"€ {ltv_value:,.0f}".replace(",", ".")) if pd.notna(ltv_value) else k5.metric("LTV (medio)","—")
k6.metric("Net New MRR (YTD)", f"€ {ytd.get('net_new_ytd', np.nan):,.0f}".replace(",", ".")) if ytd else k6.metric("Net New MRR (YTD)","—")
k7.metric("Growth YTD", f"{ytd.get('growth_ytd', np.nan):.1f}%") if ytd else k7.metric("Growth YTD","—")
k8.metric("GRR YTD", f"{ytd.get('grr_ytd', np.nan):.1f}%") if ytd else k8.metric("GRR YTD","—")
k9.metric("NRR YTD", f"{ytd.get('nrr_ytd', np.nan):.1f}%") if ytd else k9.metric("NRR YTD","—")

st.divider()

# --------- Gráfico 1: Evolución de MRR ---------
st.subheader("Evolución de MRR")
line_mrr = alt.Chart(filt).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Mes'),
    y=alt.Y('Total MRR (€):Q', title='MRR (€)'),
    tooltip=['Date:T','Total MRR (€):Q','New MRR (€):Q','Expansion MRR (inferred €):Q','Churned MRR (€):Q','Downgraded MRR (inferred €):Q']
).properties(height=300)
st.altair_chart(line_mrr, use_container_width=True)

# --------- Gráfico 2: Net New MRR (con filtros de componentes) ---------
st.subheader("NET NEW MRR por componentes")
components = {
    "New MRR (€)": st.checkbox("New", value=True),
    "Expansion MRR (inferred €)": st.checkbox("Expansion (inferred)", value=True),
    "Churned MRR (€)": st.checkbox("Churned", value=True),
    "Downgraded MRR (inferred €)": st.checkbox("Downgrade (inferred)", value=True)
}

stack_df = filt.melt(id_vars=["Date"], value_vars=[k for k,v in components.items() if v],
                     var_name="Tipo", value_name="€")

bars = alt.Chart(stack_df).mark_bar().encode(
    x=alt.X('Date:T', title='Mes'),
    y=alt.Y('sum(€):Q', title='€'),
    color=alt.Color('Tipo:N'),
    tooltip=['Date:T','Tipo:N','€:Q']
).properties(height=300)
st.altair_chart(bars, use_container_width=True)

st.divider()

# --------- Tabla por plan (filtros de año aplican) ---------
plans = comp[comp["Date"].dt.year.isin(sel_years)].copy() if sel_years else comp.copy()
last_by_plan = plans.sort_values("Date").groupby("Plan").tail(1)
tot_mrr = last_by_plan["MRR (€)"].sum()
table = (plans.groupby("Plan", as_index=False)
            .agg({
                "Active Customers":"last",
                "MRR (€)":"last",
                "New MRR (€)":"sum",
                "Expansion MRR (inferred €)":"sum",
                "Churned MRR (€)":"sum",
                "Downgraded MRR (inferred €)":"sum"
            }))
table["ARR (€)"] = table["MRR (€)"] * 12
table["Mix %"] = (table["MRR (€)"] / tot_mrr * 100).replace([np.inf,-np.inf], np.nan)
table = table.merge(df_prices[["Plan","Multiple (x ARR)"]], on="Plan", how="left")
table = table.rename(columns={
    "Active Customers":"Activos",
    "MRR (€)":"MRR (€)",
    "ARR (€)":"ARR (€)",
    "New MRR (€)":"New MRR (€)",
    "Expansion MRR (inferred €)":"Expansión MRR (€)",
    "Churned MRR (€)":"Churned MRR (€)",
    "Downgraded MRR (inferred €)":"Downgraded MRR (€)",
    "Multiple (x ARR)":"Múltiplo plan (x ARR)"
})

st.subheader("Desglose por plan")
st.dataframe(table.sort_values("MRR (€)", ascending=False), use_container_width=True)

st.divider()

# --------- Cohortes (total y por plan) ---------
st.subheader("Cohortes (aprox. FIFO) — total y por plan")
cohort_total, cohort_plan = monthly_fifo_cohorts(df_data[["Date","Plan","New Customers","Lost Customers"]])

cols = st.columns(2)
with cols[0]:
    st.markdown("**Cohortes — Total (retención %)**")
    if cohort_total.empty:
        st.info("No hay datos suficientes para cohortes.")
    else:
        st.dataframe(cohort_total.style.format('{:.0f}'), use_container_width=True, height=400)

with cols[1]:
    st.markdown("**Cohortes — Por plan (retención %)**")
    if cohort_plan.empty:
        st.info("No hay datos suficientes para cohortes por plan.")
    else:
        st.dataframe(cohort_plan.style.format('{:.0f}'), use_container_width=True, height=400)

st.divider()

# --------- Valoración ---------
st.subheader("Valoración — total y por plan")

# Total con múltiplo por sector
mrr_last = mrr_now
arr_last = arr_now
base_multiples = {
    "Horizontal SaaS": 10, "Vertical SaaS": 9, "PLG": 12, "Enterprise": 8,
    "Fintech SaaS": 12, "Health SaaS": 9, "DevTools": 10, "Otro": 8
}
base_mult = base_multiples.get(sector, 10)
mult_slider = st.slider("Múltiplo xARR (ajustable según sector)", 4.0, 25.0, float(base_mult), 0.5)
valuation_total = arr_last * mult_slider if pd.notna(arr_last) else np.nan

st.metric("Valoración total (sector)", f"€ {valuation_total:,.0f}".replace(",", "."))

# Por plan usando múltiplo de la hoja Prices
per_plan_val = table[["Plan","ARR (€)","Múltiplo plan (x ARR)"]].copy()
per_plan_val["Valoración (€)"] = per_plan_val["ARR (€)"] * per_plan_val["Múltiplo plan (x ARR)"]
st.dataframe(per_plan_val.sort_values("Valoración (€)", ascending=False), use_container_width=True)

# --------- Simulador 3–5 años ---------
st.subheader("Simulador de valoración (3–5 años)")
c1, c2, c3 = st.columns(3)
with c1:
    sim_arr0 = st.number_input("ARR actual (€)", value=float(arr_last) if pd.notna(arr_last) else 0.0, step=1000.0, format="%.2f")
with c2:
    sim_growth_m = st.slider("Crecimiento mensual (%)", 0.0, 30.0, 5.0, 0.1)
with c3:
    sim_churn_m = st.slider("Churn mensual (%)", 0.0, 15.0, 2.0, 0.1)

net_g = (sim_growth_m - sim_churn_m) / 100.0
years = [3,4,5]
proj = []
for Y in years:
    arrY = sim_arr0 * ((1 + net_g) ** (12 * Y))
    # Ajuste del múltiplo por crecimiento/churn (opcional, simple):
    growth_a = ( (1+sim_growth_m/100.0)**12 - 1 )
    churn_a  = ( (1+sim_churn_m/100.0)**12 - 1 )
    mult_adj = mult_slider * (1 + growth_a/0.40) * (1 - churn_a/0.20)
    mult_adj = max(mult_slider*0.5, min(mult_adj, mult_slider*2.0))
    valuationY = arrY * mult_slider
    valuationY_adj = arrY * mult_adj
    proj.append((f"{Y} años", arrY, valuationY, valuationY_adj, mult_adj))

df_proj = pd.DataFrame(proj, columns=["Horizonte","ARR proyectado (€)","Valor (múltiplo fijo)","Valor (ajustado crecimiento/churn)","Múltiplo ajustado"])
st.dataframe(df_proj.style.format({
    "ARR proyectado (€)": "€ {:,.0f}".format,
    "Valor (múltiplo fijo)": "€ {:,.0f}".format,
    "Valor (ajustado crecimiento/churn)": "€ {:,.0f}".format,
    "Múltiplo ajustado": "{:.2f}".format
}), use_container_width=True)

st.success("Consejo: ajusta el **margen bruto** para un LTV realista y rellena la pestaña **CAC** para obtener el ratio LTV/CAC.")
