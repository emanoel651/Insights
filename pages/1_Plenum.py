import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------------------------------------------
# Configuração da página
# -------------------------------------------------------------------
st.set_page_config(page_title="Plenum", layout="wide")
st.title("RELATÓRIO COMERCIAL — PLENUM")

# -------------------------------------------------------------------
# Debug de filesystem
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# lista o que existe nesse diretório
try:
    entries = [p.name for p in ROOT.iterdir()]
except Exception as e:
    entries = [f"Erro ao listar diretório: {e}"]

st.write("🗂️ ROOT do app:", str(ROOT))
st.write("📄 Arquivos em ROOT:", entries)

# caminho esperado para o Excel
DATA_FILE = ROOT / "Plenum_2024-2025_ordenado.xlsx"
st.write("🔍 Procurando o arquivo em:", str(DATA_FILE))

# -------------------------------------------------------------------
# Função de carregamento
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_plenum() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"❌ Arquivo não encontrado em:\n{DATA_FILE}")
        return pd.DataFrame()
    df = pd.read_excel(DATA_FILE)
    return df

# carrega e para se falhar
df = load_plenum()
if df.empty:
    st.stop()

# === Resumo de Status ===
status_counts = df["Status"].value_counts()
total   = int(status_counts.sum())
cancel  = int(status_counts.get("Cancelada", 0))
normal  = int(status_counts.get("Normal",   0))
pct_cancel = (cancel / total * 100) if total else 0

st.subheader("Resumo de Status")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Notas",     f"{total}")
col2.metric("Ativa",              f"{normal}")
col3.metric("Cancelada",          f"{cancel}")
col4.metric("% Cancelada",        f"{pct_cancel:.2f}%")

# === Evolução de Status por Mês ===
st.subheader("Evolução de Status por Mês")
stats_time = (
    df.groupby([df["Emissão"].dt.to_period("M"), "Status"])
      .size()
      .reset_index(name="Quantidade")
)
stats_time["Emissão"] = stats_time["Emissão"].astype(str)
fig1 = px.line(
    stats_time, x="Emissão", y="Quantidade", color="Status",
    markers=True,
    labels={"Quantidade":"Nº Registros", "Emissão":"Mês"}
)
st.plotly_chart(fig1, use_container_width=True)

# === Evolução do Valor Cancelado por Mês ===
st.subheader("Evolução de Valor Cancelado por Mês")
cancel_time = (
    df[df["Status"]=="Cancelada"]
      .groupby(df["Emissão"].dt.to_period("M"))["Valor_Servicos"]
      .sum()
      .reset_index()
)
cancel_time["Emissão"] = cancel_time["Emissão"].astype(str)
fig2 = px.line(
    cancel_time, x="Emissão", y="Valor_Servicos",
    markers=True,
    labels={"Valor_Servicos":"Vendas Canceladas (R$)", "Emissão":"Mês"}
)
st.plotly_chart(fig2, use_container_width=True)

# === KPIs de Vendas ===
st.subheader("KPIs de Vendas")
total_vendas = df["Valor_Servicos"].sum()
v1 = df[df["Emissão"].dt.month <= 6]["Valor_Servicos"].sum()
v2 = df[df["Emissão"].dt.month >= 7]["Valor_Servicos"].sum()
k1, k2, k3 = st.columns(3)
k1.metric("Total de Vendas",   f"R$ {total_vendas:,.2f}")
k2.metric("1º Semestre",       f"R$ {v1:,.2f}")
k3.metric("2º Semestre",       f"R$ {v2:,.2f}")

# === Evolução de Vendas por Mês ===
st.subheader("Evolução de Vendas por Mês")
sales_time = (
    df.groupby(df["Emissão"].dt.to_period("M"))["Valor_Servicos"]
      .sum()
      .reset_index()
)
sales_time["Emissão"] = sales_time["Emissão"].astype(str)
fig3 = px.line(
    sales_time, x="Emissão", y="Valor_Servicos",
    markers=True,
    labels={"Valor_Servicos":"Vendas (R$)", "Emissão":"Mês"}
)
st.plotly_chart(fig3, use_container_width=True)

# === Top 10 Cidades ===
st.subheader("Top 10 Cidades")
cid_top = (
    df.groupby("Cidade")["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig4 = px.bar(
    cid_top.head(10), x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s",
    labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig4, use_container_width=True)

# === Top 10 Mesorregiões ===
st.subheader("Top 10 Mesorregiões")
meso_top = (
    df.groupby("Mesorregiao")["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig5 = px.bar(
    meso_top.head(10), x="Valor_Servicos", y="Mesorregiao",
    orientation="h", text_auto=".2s",
    labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig5, use_container_width=True)

# === Top 10 Microrregiões ===
st.subheader("Top 10 Microrregiões")
micro_top = (
    df.groupby("Microrregiao")["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig6 = px.bar(
    micro_top.head(10), x="Valor_Servicos", y="Microrregiao",
    orientation="h", text_auto=".2s",
    labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig6, use_container_width=True)

# === Gráfico Adicional: Top 10 Cidades com Mesorregião & Microrregião ===
st.subheader("Top 10 Cidades por Mesorregião & Microrregião")
city_meso_micro = (
    df.groupby(["Mesorregiao","Microrregiao","Cidade"])["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
      .head(10)
)
fig_extra = px.bar(
    city_meso_micro,
    x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s",
    hover_data=["Mesorregiao","Microrregiao"],
    labels={"Valor_Servicos":"R$","Cidade":"Cidade"}
)
st.plotly_chart(fig_extra, use_container_width=True)

# === Top 10 Cidades por Mesorregião ===
st.subheader("Top 10 Cidades por Mesorregião")
city_by_meso = (
    df.groupby(["Mesorregiao","Cidade"])["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
      .head(10)
)
fig7 = px.bar(
    city_by_meso,
    x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s",
    color="Mesorregiao",
    labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig7, use_container_width=True)

# === Top 10 Cidades por Microrregião ===
st.subheader("Top 10 Cidades por Microrregião")
city_by_micro = (
    df.groupby(["Microrregiao","Cidade"])["Valor_Servicos"]
      .sum()
      .reset_index()
      .sort_values("Valor_Servicos", ascending=False)
      .head(10)
)
fig8 = px.bar(
    city_by_micro,
    x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s",
    color="Microrregiao",
    labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig8, use_container_width=True)

# === Tabela de Cidades com Faturamento ===
st.subheader("Tabela de Cidades com Faturamento")
table = cid_top.head(20).copy()
table["Valor_Servicos"] = table["Valor_Servicos"].apply(
    lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
)
st.dataframe(table, use_container_width=True)

# === Destaques Finais ===
st.markdown("---")
st.markdown(f"**🌿 Mesorregião top:**  {meso_top.iloc[0]['Mesorregiao']}  —  R$ {meso_top.iloc[0]['Valor_Servicos']:,.2f}")
st.markdown(f"**🌺 Microrregião top:** {micro_top.iloc[0]['Microrregiao']} — R$ {micro_top.iloc[0]['Valor_Servicos']:,.2f}")
st.markdown(f"**🏙️ Cidade top:**       {cid_top.iloc[0]['Cidade']}   —   R$ {cid_top.iloc[0]['Valor_Servicos']:,.2f}")
