import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Insights", layout="wide")
st.title("💡 Insights Gerais — Plenum + Instituto")

# ------------------------------------------------------------------------------
# Cached data loader
# ------------------------------------------------------------------------------
@st.cache_data
def load_data(filename: str) -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent
    path = base / filename
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    valor_cols = [c for c in df.columns if "Valor" in c]
    if valor_cols:
        df = df.rename(columns={valor_cols[0]: "Valor_Servicos"})
    df["Emissão"] = pd.to_datetime(df["Emissão"], errors="coerce")
    df["Valor_Servicos"] = (
        df["Valor_Servicos"].astype(str)
          .str.replace(".", "", regex=False)
          .str.replace(",", ".", regex=False)
    ).astype(float)
    df = df.dropna(subset=["Emissão", "Valor_Servicos", "Mesorregiao", "Microrregiao", "Cidade"])
    return df

# Load datasets
_df_inst   = load_data("Institulo_2024-2025_ordenado.xlsx")
df_plenum = load_data("Plenum_2024-2025_ordenado.xlsx")
df_all    = pd.concat([_df_inst, df_plenum], ignore_index=True)

# Metric: total sales
st.metric("💰 Total de Vendas", f"R$ {df_all['Valor_Servicos'].sum():,.2f}")

# ------------------------------------------------------------------------------
# 1) Monthly sales evolution
# ------------------------------------------------------------------------------
df_time = (
    df_all
      .groupby(df_all["Emissão"].dt.to_period("M"))["Valor_Servicos"]
      .sum()
      .reset_index()
)
df_time["Emissão"] = df_time["Emissão"].dt.strftime("%Y-%m")
fig_time = px.line(
    df_time, x="Emissão", y="Valor_Servicos",
    title="Evolução Mensal de Vendas (Plenum + Instituto)",
    labels={"Emissão":"Mês", "Valor_Servicos":"Vendas (R$)"}
)
st.plotly_chart(fig_time, use_container_width=True)

# Helper: top N

def top_n(df, by, n=10):
    return (
        df
          .groupby(by)["Valor_Servicos"]
          .sum()
          .reset_index()
          .sort_values("Valor_Servicos", ascending=False)
          .head(n)
    )

# ------------------------------------------------------------------------------
# 2) Top 10 by region
# ------------------------------------------------------------------------------
st.subheader("Top 10 Mesorregiões")
df_meso = top_n(df_all, "Mesorregiao")
fig_meso = px.bar(
    df_meso, x="Valor_Servicos", y="Mesorregiao", orientation="h",
    title="Top 10 Mesorregiões por Vendas Totais",
    labels={"Valor_Servicos":"R$", "Mesorregiao":"Mesorregião"}
)
st.plotly_chart(fig_meso, use_container_width=True)

st.subheader("Top 10 Microrregiões")
df_micro = top_n(df_all, "Microrregiao")
fig_micro = px.bar(
    df_micro, x="Valor_Servicos", y="Microrregiao", orientation="h",
    title="Top 10 Microrregiões por Vendas Totais",
    labels={"Valor_Servicos":"R$", "Microrregiao":"Microrregião"}
)
st.plotly_chart(fig_micro, use_container_width=True)

st.subheader("Top 10 Cidades")
df_city = top_n(df_all, "Cidade")
fig_city = px.bar(
    df_city, x="Valor_Servicos", y="Cidade", orientation="h",
    title="Top 10 Cidades por Vendas Totais",
    labels={"Valor_Servicos":"R$", "Cidade":"Cidade"}
)
st.plotly_chart(fig_city, use_container_width=True)

# ------------------------------------------------------------------------------
# 3) Summary by region for ML
# ------------------------------------------------------------------------------
df_summary = (
    df_all
      .groupby(["Mesorregiao","Microrregiao"])["Valor_Servicos"]
      .agg(
          Valor_Servicos_Total="sum",
          Valor_Servicos_Medio="mean",
          Numero_Servicos="count"
      )
      .reset_index()
)

# ------------------------------------------------------------------------------
# 4) Clustering analysis
# ------------------------------------------------------------------------------
st.subheader("Clustering de Regiões")
le_meso  = LabelEncoder().fit(df_summary["Mesorregiao"])
le_micro = LabelEncoder().fit(df_summary["Microrregiao"])
df_summary["meso_enc"]  = le_meso.transform(df_summary["Mesorregiao"])
df_summary["micro_enc"] = le_micro.transform(df_summary["Microrregiao"])

X_cluster = df_summary[[
    "Valor_Servicos_Total",
    "Valor_Servicos_Medio",
    "Numero_Servicos",
    "meso_enc",
    "micro_enc"
]]
scaler   = StandardScaler().fit(X_cluster)
X_scaled = scaler.transform(X_cluster)

# K-Means
kmeans  = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
labels_km = kmeans.labels_
sil_km    = silhouette_score(X_scaled, labels_km)
st.metric("Silhouette K-Means", f"{sil_km:.3f}")

pca     = PCA(n_components=2).fit(X_scaled)
coords  = pca.transform(X_scaled)
df_viz  = pd.DataFrame({
    "PC1": coords[:,0],
    "PC2": coords[:,1],
    "Cluster": labels_km.astype(str),
    "Mesorregiao": df_summary["Mesorregiao"],
    "Microrregiao": df_summary["Microrregiao"]
})
fig_km = px.scatter(
    df_viz, x="PC1", y="PC2",
    color="Cluster",
    hover_data=["Mesorregiao","Microrregiao"],
    title="Clusters K‑Means (PCA 2D)"
)
st.plotly_chart(fig_km, use_container_width=True)

# DBSCAN
labels_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
clusters_db = set(labels_db)
n_clusters_db = len(clusters_db - {-1})
if n_clusters_db >= 2:
    mask = labels_db != -1
    sil_db = silhouette_score(X_scaled[mask], labels_db[mask])
    st.metric("Silhouette DBSCAN (sem ruído)", f"{sil_db:.3f}")
else:
    st.warning("DBSCAN não gerou clusters suficientes para calcular Silhouette.")
st.subheader("Contagem de Clusters DBSCAN")
st.dataframe(
    pd.Series(labels_db)
      .value_counts()
      .rename_axis("Cluster")
      .reset_index(name="Count")
)

# ------------------------------------------------------------------------------
# 5) Regression feature importance
# ------------------------------------------------------------------------------
st.subheader("Importância de Features (RandomForest)")
X_reg = df_summary[["Valor_Servicos_Medio","Numero_Servicos","meso_enc","micro_enc"]]
y_reg = df_summary["Valor_Servicos_Total"]
rf      = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_reg, y_reg)
feat_imp = pd.Series(rf.feature_importances_, index=X_reg.columns).sort_values()
fig_imp = px.bar(
    feat_imp, x=feat_imp.values, y=feat_imp.index, orientation="h",
    title="Importância das Features para Valor Total de Serviços",
    labels={"x":"Importância","y":"Feature"}
)
st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# 6) “Vale Investir” – modelagem e gráficos
# ------------------------------------------------------------------------------
# Define o threshold no percentil 70%
threshold = df_summary["Valor_Servicos_Total"].quantile(0.7)
df_summary["vale_investir"] = (df_summary["Valor_Servicos_Total"] >= threshold).astype(int)
# Treina regressão logística
X = df_summary[["Valor_Servicos_Medio", "Numero_Servicos"]]
y = df_summary["vale_investir"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
clf = LogisticRegression(random_state=42).fit(X_train, y_train)
# Probabilidades para todas as regiões
df_summary["proba_v_investir"] = clf.predict_proba(X)[:, 1]

# Gráfico Mesorregiões “vale investir”
st.subheader("📈 Mesorregiões que Valem Investir")
df_meso_inv = (
    df_summary[df_summary["vale_investir"] == 1]
      .groupby("Mesorregiao")["proba_v_investir"]
      .mean()
      .reset_index()
      .sort_values("proba_v_investir", ascending=False)
)
df_meso_inv["prob_pct"] = df_meso_inv["proba_v_investir"] * 100
fig_meso_inv = px.bar(
    df_meso_inv, x="prob_pct", y="Mesorregiao", orientation="h",
    title="Probabilidade Média (%) de “Vale Investir” – Mesorregiões",
    labels={"prob_pct":"Probabilidade (%)", "Mesorregiao":"Mesorregião"},
    text="prob_pct"
)
fig_meso_inv.update_traces(texttemplate="%{text:.1f}%")
st.plotly_chart(fig_meso_inv, use_container_width=True)

# Gráfico Microrregiões “vale investir”
st.subheader("📊 Microrregiões que Valem Investir")
df_micro_inv = (
    df_summary[df_summary["vale_investir"] == 1]
      .loc[:, ["Microrregiao", "proba_v_investir"]]
      .sort_values("proba_v_investir", ascending=False)
      .reset_index(drop=True)
)
df_micro_inv["prob_pct"] = df_micro_inv["proba_v_investir"] * 100
fig_micro_inv = px.bar(
    df_micro_inv, x="prob_pct", y="Microrregiao", orientation="h",
    title="Probabilidade Média (%) de “Vale Investir” – Microrregiões",
    labels={"prob_pct":"Probabilidade (%)", "Microrregiao":"Microrregião"},
    text="prob_pct"
)
fig_micro_inv.update_traces(texttemplate="%{text:.1f}%")
st.plotly_chart(fig_micro_inv, use_container_width=True)

# Exibição de métricas do modelo
st.subheader("Classificação “Vale Investir” vs “Não Vale”")
st.write(f"Threshold (70º percentil): R$ {threshold:,.2f}")
y_pred = clf.predict(X_test)
st.write(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
st.markdown("**Relatório de Classificação:**")
st.text(classification_report(y_test, y_pred, target_names=["Não Vale", "Vale"]))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
labels_cm = np.array([[f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm_norm, annot=labels_cm, fmt="", cmap="Blues", cbar=False, linewidths=0.5, annot_kws={"size":14}, ax=ax)
ax.set_title("Matriz de Confusão (count + %)", fontsize=16)
ax.set_xlabel("Classe Predita", fontsize=14)
ax.set_ylabel("Classe Verdadeira", fontsize=14)
ax.set_xticklabels(["Não Vale","Vale"], rotation=0)
ax.set_yticklabels(["Não Vale","Vale"], rotation=0)
plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------------------------------------------
# 7) Conclusões finais
# ------------------------------------------------------------------------------
# Cálculo de crescimento
first_val = df_time["Valor_Servicos"].iloc[0]
last_val  = df_time["Valor_Servicos"].iloc[-1]
growth    = (last_val - first_val) / first_val * 100 if first_val else 0
# Crescimento médio por região
meso_period = (
    df_all
      .groupby(["Mesorregiao", df_all["Emissão"].dt.to_period("M")])["Valor_Servicos"]
      .sum().reset_index()
)
growths_meso = []
for meso in meso_period["Mesorregiao"].unique():
    ts = meso_period[meso_period["Mesorregiao"]==meso].sort_values("Emissão")["Valor_Servicos"]
    if len(ts)>1 and ts.iloc[0]!=0:
        growths_meso.append((ts.iloc[-1]-ts.iloc[0])/ts.iloc[0]*100)
avg_growth_meso = np.mean(growths_meso) if growths_meso else 0
micro_period = (
    df_all
      .groupby(["Microrregiao", df_all["Emissão"].dt.to_period("M")])["Valor_Servicos"]
      .sum().reset_index()
)
growths_micro=[]
for micro in micro_period["Microrregiao"].unique():
    ts= micro_period[micro_period["Microrregiao"]==micro].sort_values("Emissão")["Valor_Servicos"]
    if len(ts)>1 and ts.iloc[0]!=0:
        growths_micro.append((ts.iloc[-1]-ts.iloc[0])/ts.iloc[0]*100)
avg_growth_micro=np.mean(growths_micro) if growths_micro else 0

st.markdown("---")
st.markdown(f"""
<div style="font-size:1.3rem; line-height:1.5;">
  <h2 style="font-size:2rem; margin-bottom:0.5rem;">📝 Conclusões e Pontos Positivos</h2>
  <ul>  
    <li><strong>Crescimento geral</strong>: ↑ {growth:.1f}% entre {df_time['Emissão'].iloc[0]} e {df_time['Emissão'].iloc[-1]}.</li>
    <li><strong>Crescimento médio Mesorregiões</strong>: ↑ {avg_growth_meso:.1f}% no período.</li>
    <li><strong>Crescimento médio Microrregiões</strong>: ↑ {avg_growth_micro:.1f}% no período.</li>
    <li><strong>Recomendação</strong>: focar em regiões com alta probabilidade de investimento e crescimento consistente.</li>
    <li><strong>Recomendação</strong>: As mesoregiões que mais vale apena investir, são Araraquara com 99,47% e Sul/Sudoeste de Minas com 93,99%
     <li><strong>Recomendação</strong>: Mesmo com auumento de 138%,4%, microregião continua com as taxas de sucesso maior que mesoregião.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
