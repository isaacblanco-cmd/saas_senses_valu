# SaaS Valuation & MRR Dashboard

Un panel en **Streamlit** que calcula métricas SaaS (MRR, ARR, Net New, GRR/NRR, cohortes FIFO aproximadas) y una **valoración** por múltiplos. Permite filtrar por año y desglosa por plan. También calcula **LTV/CAC** usando la pestaña **CAC** del Excel.

## Estructura de Excel
Se requieren las hojas **Prices** y **Data**. Opcional **CAC**.
- **Prices**: columnas `Plan`, `Price MRR (€)`, `Price ARR (€)`, `Multiple (x ARR)`.
- **Data**: columnas `Date`, `Plan`, `New Customers`, `Lost Customers`, `Active Customers (optional)`, `Real MRR (optional €)`.
- **CAC** (opcional): columnas `Date`, `Sales & Marketing Spend (€)`, y una de `New Customers` o `New Customers (from Data)`.

> Te dejamos un archivo de ejemplo con la pestaña **CAC** ya creada: `SaaS_Final_Template_COMPLETO_with_CAC.xlsx`.

## Cómo ejecutar
```bash
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10–3.12 recomendado.

## Notas de cálculo
- **MRR real** se toma de `Real MRR (optional €)` si está rellena; si no, se estima con `Active Customers (optional) * Price MRR`.
- **Expansión/Downgrade (inferred)**: a partir del residual de `ΔMRR - New + Churn`.
- **GRR%** = 1 − (Churned + Downgrade) / Start MRR.
- **NRR%** = 1 + (Expansión − Churned − Downgrade) / Start MRR.
- **Quick Ratio** = (New + Expansión) / (Churned + Downgrade).
- **LTV** ≈ ARPU × Margen / Churn mensual.
- **Valoración** total = ARR × múltiplo (selector de sector + slider). **Por plan** se usa `Multiple (x ARR)` de **Prices**.
- **Cohortes**: aproximación FIFO con datos agregados mensuales (sin nivel cliente).

## Licencia
MIT.
