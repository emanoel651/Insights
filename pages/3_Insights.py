# 🔧 Imports e Configurações
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Página e título
# ----------------------------------------------------------------------
st.set_page_config(page_title="Insights", layout="wide")
st.title("💡 Insights Gerais — Plenum + Instituto")

# ----------------------------------------------------------------------
# Função para carregar dados com cache
# ----------------------------------------------------------------------
@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    try:
        base = Path(__file__).resolve().parent.parent
    except NameError:
        base = Path.cwd()
    path = base / filename
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    valor_cols = [c for c in df.columns if "Valor" in c]
    if valor_cols:
        df = df.rename(columns={valor_cols[0]: "Valor_Servicos"})

    df["Emissão"] = pd.to_datetime(df["Emissão"], errors="coerce")

    if df["Valor_Servicos"].dtype == object:
        df["Valor_Servicos"] = (
            df["Valor_Servicos"]
              .str.replace(r"\.", "", regex=True)
              .str.replace(",", ".", regex=False)
        )
    df["Valor_Servicos"] = pd.to_numeric(df["Valor_Servicos"], errors="coerce")
    df = df.dropna(subset=["Emissão", "Valor_Servicos", "Mesorregiao", "Microrregiao", "Cidade"])
    return df

# ----------------------------------------------------------------------
# Carregamento e junção dos dados
# ----------------------------------------------------------------------
_df_inst  = load_data("Institulo_2024-2025_ordenado.xlsx")
df_plenum = load_data("Plenum_2024-2025_ordenado.xlsx")
df_all    = pd.concat([_df_inst, df_plenum], ignore_index=True)

# ----------------------------------------------------------------------
# Métricas de topo
# ----------------------------------------------------------------------
st.metric("💰 Total de Vendas", f"R$ {df_all['Valor_Servicos'].sum():,.2f}")

# Evolução mensal
df_time = df_all.groupby(df_all["Emissão"].dt.to_period("M"))["Valor_Servicos"].sum().reset_index()
df_time["Emissão"] = df_time["Emissão"].dt.strftime("%Y-%m")
st.plotly_chart(px.line(df_time, x="Emissão", y="Valor_Servicos",
                        title="Evolução Mensal de Vendas",
                        labels={"Emissão":"Mês", "Valor_Servicos":"Vendas (R$)"}),
                use_container_width=True)

# ----------------------------------------------------------------------
# Função auxiliar para top N
# ----------------------------------------------------------------------
def top_n(df, by, n=10):
    return df.groupby(by)["Valor_Servicos"].sum().reset_index().sort_values("Valor_Servicos", ascending=False).head(n)

# Top Mesorregiões
st.subheader("Top 10 Mesorregiões")
st.plotly_chart(px.bar(top_n(df_all, "Mesorregiao"), x="Valor_Servicos", y="Mesorregiao", orientation="h",
                       title="Top Mesorregiões", labels={"Valor_Servicos":"R$", "Mesorregiao":"Mesorregião"}),
                use_container_width=True)

# Top Microrregiões
st.subheader("Top 10 Microrregiões")
st.plotly_chart(px.bar(top_n(df_all, "Microrregiao"), x="Valor_Servicos", y="Microrregiao", orientation="h",
                       title="Top Microrregiões", labels={"Valor_Servicos":"R$", "Microrregiao":"Microrregião"}),
                use_container_width=True)

# Top Cidades
st.subheader("Top 10 Cidades")
st.plotly_chart(px.bar(top_n(df_all, "Cidade"), x="Valor_Servicos", y="Cidade", orientation="h",
                       title="Top Cidades", labels={"Valor_Servicos":"R$", "Cidade":"Cidade"}),
                use_container_width=True)

# ----------------------------------------------------------------------
# Agrupamento de dados para análise
# ----------------------------------------------------------------------
df_summary = df_all.groupby(["Mesorregiao","Microrregiao"])["Valor_Servicos"].agg(
    Valor_Servicos_Total="sum",
    Valor_Servicos_Medio="mean",
    Numero_Servicos="count"
).reset_index()

# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------
st.subheader("Clustering de Regiões")
le_meso  = LabelEncoder().fit(df_summary["Mesorregiao"])
le_micro = LabelEncoder().fit(df_summary["Microrregiao"])
df_summary["meso_enc"]  = le_meso.transform(df_summary["Mesorregiao"])
df_summary["micro_enc"] = le_micro.transform(df_summary["Microrregiao"])

X_cluster = df_summary[["Valor_Servicos_Total", "Valor_Servicos_Medio", "Numero_Servicos", "meso_enc", "micro_enc"]]
X_scaled = StandardScaler().fit_transform(X_cluster)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df_summary["cluster_km"] = kmeans.labels_
st.metric("Silhouette K-Means", f"{silhouette_score(X_scaled, kmeans.labels_):.3f}")

pca_coords = PCA(n_components=2).fit_transform(X_scaled)
st.plotly_chart(px.scatter(x=pca_coords[:,0], y=pca_coords[:,1], color=df_summary["cluster_km"].astype(str),
                           title="Clusters K‑Means (PCA)", labels={"x":"PC1","y":"PC2"},
                           hover_data={"Mesorregiao":df_summary["Mesorregiao"],
                                       "Microrregiao":df_summary["Microrregiao"]}),
                use_container_width=True)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels_db = dbscan.labels_
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
if n_clusters_db >= 2:
    sil_db = silhouette_score(X_scaled[labels_db != -1], labels_db[labels_db != -1])
    st.metric("Silhouette DBSCAN", f"{sil_db:.3f}")
else:
    st.warning("DBSCAN não gerou clusters suficientes para silhouette.")
st.dataframe(pd.Series(labels_db).value_counts().rename_axis("Cluster").reset_index(name="Count"))

# ----------------------------------------------------------------------
# Random Forest: Importância das variáveis
# ----------------------------------------------------------------------
st.subheader("Importância das Features")
X_rf = df_summary[["Valor_Servicos_Medio", "Numero_Servicos", "meso_enc", "micro_enc"]]
y_rf = df_summary["Valor_Servicos_Total"]
feat_imp = pd.Series(RandomForestRegressor(n_estimators=100, random_state=42).fit(X_rf, y_rf).feature_importances_, index=X_rf.columns).sort_values()
st.plotly_chart(px.bar(feat_imp, x=feat_imp.values, y=feat_imp.index, orientation="h",
                       title="Importância das Features", labels={"x":"Importância", "y":"Feature"}),
                use_container_width=True)
