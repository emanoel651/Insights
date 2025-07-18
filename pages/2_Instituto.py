import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re

# ─── Configuração da página ──────────────────────────────────
st.set_page_config(page_title="Instituto", layout="wide")
st.title("RELATÓRIO COMERCIAL — INSTITUTO")

# ─── Função de carregamento ──────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    file = root / "Instituto_2024-2025_ordenado.xlsx"
    if not file.exists():
        st.error(f"❌ Arquivo não encontrado: {file}")
        return pd.DataFrame()

    # 1) Lê o Excel
    df = pd.read_excel(file)

    # 2) Renomeia a primeira coluna que contenha "valor" para "Valor_Servicos"
    valor_cols = [c for c in df.columns if re.search(r"valor", c, re.IGNORECASE)]
    if not valor_cols:
        st.error("❌ Não encontrei nenhuma coluna com 'valor' no nome.")
        return pd.DataFrame()
    df = df.rename(columns={valor_cols[0]: "Valor_Servicos"})

    # 3) Converte tipos
    df["Emissão"] = pd.to_datetime(df["Emissão"], errors="coerce")
    df["Valor_Servicos"] = pd.to_numeric(df["Valor_Servicos"], errors="coerce")

    # 4) Remove linhas com dados faltantes nas colunas essenciais
    required = [
        "Cidade", "Estado", "Região",
        "Mesorregiao", "Microrregiao",
        "Status", "Emissão", "Valor_Servicos"
    ]
    df = df.dropna(subset=required)

    return df

# ─── Carrega os dados ───────────────────────────────────────
df = load_data()
if df.empty:
    st.stop()

# ─── Resumo de Status ────────────────────────────────────────
status_counts = df["Status"].value_counts()
total      = int(status_counts.sum())
cancel     = int(status_counts.get("Cancelada", 0))
normal     = int(status_counts.get("Normal",   0))
pct_cancel = cancel / total * 100 if total else 0

st.subheader("Resumo de Status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de Registros", f"{total}")
c2.metric("Normal",            f"{normal}")
c3.metric("Cancelada",         f"{cancel}")
c4.metric("% Cancelada",       f"{pct_cancel:.2f}%")

# ─── Evolução de Status por Mês ─────────────────────────────
st.subheader("Evolução de Status por Mês")
stats_time = (
    df.groupby([df["Emissão"].dt.to_period("M"), "Status"])
      .size().reset_index(name="Quantidade")
)
stats_time["Emissão"] = stats_time["Emissão"].astype(str)
fig1 = px.line(
    stats_time, x="Emissão", y="Quantidade", color="Status",
    markers=True, labels={"Quantidade":"Nº Registros", "Emissão":"Mês"}
)
st.plotly_chart(fig1, use_container_width=True, key="ins_status")

# ─── Evolução de Valor Cancelado por Mês ────────────────────
st.subheader("Evolução de Valor Cancelado por Mês")
cancel_time = (
    df[df["Status"]=="Cancelada"]
      .groupby(df["Emissão"].dt.to_period("M"))["Valor_Servicos"]
      .sum().reset_index()
)
cancel_time["Emissão"] = cancel_time["Emissão"].astype(str)
fig2 = px.line(
    cancel_time, x="Emissão", y="Valor_Servicos",
    markers=True,
    labels={"Valor_Servicos":"Vendas Canceladas (R$)", "Emissão":"Mês"}
)
st.plotly_chart(fig2, use_container_width=True, key="ins_cancelados")

# ─── KPIs de Vendas ─────────────────────────────────────────
st.subheader("KPIs de Vendas")
total_vendas = df["Valor_Servicos"].sum()
v1 = df[df["Emissão"].dt.month <= 6]["Valor_Servicos"].sum()
v2 = df[df["Emissão"].dt.month >= 7]["Valor_Servicos"].sum()
k1, k2, k3 = st.columns(3)
k1.metric("Total de Vendas", f"R$ {total_vendas:,.2f}")
k2.metric("1º Semestre",     f"R$ {v1:,.2f}")
k3.metric("2º Semestre",     f"R$ {v2:,.2f}")

# ─── Evolução de Vendas por Mês ─────────────────────────────
st.subheader("Evolução de Vendas por Mês")
sales_time = (
    df.groupby(df["Emissão"].dt.to_period("M"))["Valor_Servicos"]
      .sum().reset_index()
)
sales_time["Emissão"] = sales_time["Emissão"].astype(str)
fig3 = px.line(
    sales_time, x="Emissão", y="Valor_Servicos",
    markers=True, labels={"Valor_Servicos":"Vendas (R$)", "Emissão":"Mês"}
)
st.plotly_chart(fig3, use_container_width=True, key="ins_vendas")

# ─── Top 10 Cidades ──────────────────────────────────────────
st.subheader("Top 10 Cidades")
cid_top = (
    df.groupby("Cidade")["Valor_Servicos"]
      .sum().reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig4 = px.bar(
    cid_top.head(10), x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s", labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig4, use_container_width=True, key="ins_top_cidades")

# ─── Top 10 Mesorregiões ─────────────────────────────────────
st.subheader("Top 10 Mesorregiões")
meso_top = (
    df.groupby("Mesorregiao")["Valor_Servicos"]
      .sum().reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig5 = px.bar(
    meso_top.head(10), x="Valor_Servicos", y="Mesorregiao",
    orientation="h", text_auto=".2s", labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig5, use_container_width=True, key="ins_top_meso")

# ─── Top 10 Microrregiões ────────────────────────────────────
st.subheader("Top 10 Microrregiões")
micro_top = (
    df.groupby("Microrregiao")["Valor_Servicos"]
      .sum().reset_index()
      .sort_values("Valor_Servicos", ascending=False)
)
fig6 = px.bar(
    micro_top.head(10), x="Valor_Servicos", y="Microrregiao",
    orientation="h", text_auto=".2s", labels={"Valor_Servicos":"R$"}
)
st.plotly_chart(fig6, use_container_width=True, key="ins_top_micro")

# ─── Top 10 Cidades por Meso & Micro ───────────────────────
st.subheader("Top 10 Cidades por Mesorregião & Microrregião")
city_meso_micro = (
    df.groupby(["Mesorregiao","Microrregiao","Cidade"])["Valor_Servicos"]
      .sum().reset_index()
      .sort_values("Valor_Servicos", ascending=False)
      .head(10)
)
fig7 = px.bar(
    city_meso_micro, x="Valor_Servicos", y="Cidade",
    orientation="h", text_auto=".2s",
    hover_data=["Mesorregiao","Microrregiao"],
    labels={"Valor_Servicos":"R$","Cidade":"Cidade"}
)
st.plotly_chart(fig7, use_container_width=True, key="ins_extra")

# ─── Tabela de Cidades ──────────────────────────────────────
st.subheader("Tabela de Cidades com Faturamento")
table = cid_top.head(20).copy()
table["Valor_Servicos"] = table["Valor_Servicos"].apply(
    lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
)
st.dataframe(table, use_container_width=True)

# ─── Destaques Finais ───────────────────────────────────────
st.markdown("---")
st.markdown(f"**🌿 Mesorregião top:**  {meso_top.iloc[0]['Mesorregiao']} — R$ {meso_top.iloc[0]['Valor_Servicos']:,.2f}")
st.markdown(f"**🌺 Microrregião top:** {micro_top.iloc[0]['Microrregiao']} — R$ {micro_top.iloc[0]['Valor_Servicos']:,.2f}")
st.markdown(f"**🏙️ Cidade top:**       {cid_top.iloc[0]['Cidade']}   — R$ {cid_top.iloc[0]['Valor_Servicos']:,.2f}")
