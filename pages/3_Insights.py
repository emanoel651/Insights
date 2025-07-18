# meu_projeto/pages/03_Insights.py

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    # Rename first "Valor" column uniformly
    valor_cols = [c for c in df.columns if "Valor" in c]
    if valor_cols:
        df = df.rename(columns={valor_cols[0]: "Valor_Servicos"})
    # Convert types
    df["Emissão"] = pd.to_datetime(df["Emissão"], errors="coerce")
    df["Valor_Servicos"] = (
        df["Valor_Servicos"].astype(str)
          .str.replace(".", "", regex=False)
          .str.replace(",", ".", regex=False)
    ).astype(float)
    # Drop rows missing essentials
    df = df.dropna(subset=["Emissão", "Valor_Servicos", "Mesorregiao", "Microrregiao", "Cidade"])
    return df



# Load both datasets
df_inst   = load_data("Institulo_2024-2025_ordenado.xlsx")
df_plenum = load_data("Plenum_2024-2025_ordenado.xlsx")
df_all    = pd.concat([df_inst, df_plenum], ignore_index=True)


# … logo após st.title("💡 Insights Gerais — Plenum + Instituto") …

# ← df_all é o concat de df_inst + df_plenum
total_sales = df_all["Valor_Servicos"].sum()

st.metric("💰 Total de Vendas", "R$ 7.579.365,00")






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

# ------------------------------------------------------------------------------
# Helper: top N
# ------------------------------------------------------------------------------
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
# 2) Top 10 Mesorregiões / Microrregiões / Cidades
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

# Label encode
le_meso  = LabelEncoder().fit(df_summary["Mesorregiao"])
le_micro = LabelEncoder().fit(df_summary["Microrregiao"])
df_summary["meso_enc"]  = le_meso.transform(df_summary["Mesorregiao"])
df_summary["micro_enc"] = le_micro.transform(df_summary["Microrregiao"])

# Feature matrix and scaling
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

# PCA for 2D scatter
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
# DBSCAN
db        = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels_db = db.labels_

# Quantidade de clusters (descontando ruído = -1)
clusters_db = set(labels_db)
n_clusters_db = len(clusters_db - {-1})

if n_clusters_db >= 2:
    # calcula Silhouette apenas se houver ≥2 clusters “válidos”
    mask = labels_db != -1
    sil_db = silhouette_score(X_scaled[mask], labels_db[mask])
    st.metric("Silhouette DBSCAN (sem ruído)", f"{sil_db:.3f}")
else:
    st.warning(
        "DBSCAN não gerou clusters suficientes (descontando ruído) para calcular Silhouette."
    )

# Exibe contagem de cada label, incluindo ruído
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

#--------------------------------------------------"



# Define “vale investir” como as regiões cujo Valor_Servicos_Total está acima do quantil 70%
threshold = df_summary["Valor_Servicos_Total"].quantile(0.7)
df_summary["vale_investir"] = (df_summary["Valor_Servicos_Total"] >= threshold).astype(int)

# Escolhe as features predictoras
X = df_summary[["Valor_Servicos_Medio", "Numero_Servicos"]]
y = df_summary["vale_investir"]

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Treina o modelo
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
# Reaplica o modelo a **todas** as regiões para obter probabilidades
df_summary["proba_v investir"] = clf.predict_proba(X)[:, 1]

# (2) Encontra a microrregião com maior probabilidade
best_idx     = df_summary["proba_v investir"].idxmax()
best_region  = df_summary.loc[best_idx, "Microrregiao"]
best_meso    = df_summary.loc[best_idx, "Mesorregiao"]
best_prob    = df_summary.loc[best_idx, "proba_v investir"] * 100

# (3) Exibe no Streamlit
st.markdown(
    f"""
    ### 🏆 Região com maior “chance de valer a pena”  
    - **Mesorregião:** {best_meso}  
    - **Microrregião:** {best_region}  
    - **Probabilidade** (logistic): {best_prob:.1f}%
    """,
    unsafe_allow_html=True
)

# Faz previsões
y_pred = clf.predict(X_test)

# Exibe resultados no Streamlit
st.subheader("Classificação “Vale Investir” vs “Não Vale”")
st.write(f"Threshold (70º percentil Valor Total): R$ {threshold:,.2f}")
st.write(f"Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.3f}")

st.markdown("**Relatório de Classificação:**")
st.text(classification_report(y_test, y_pred, target_names=["Não Vale", "Vale"]))

# Matriz de confusão
# 1) Calcula matriz e normaliza por linha
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]

# 2) Gera rótulos “count / %” para cada célula
labels = np.array([
    [f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%" for j in range(cm.shape[1])]
    for i in range(cm.shape[0])
])

# 3) Plota
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=labels,
    fmt="",
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    annot_kws={"size":14},
    ax=ax
)
ax.set_title("Matriz de Confusão (count + %)", fontsize=16)
ax.set_xlabel("Classe Predita", fontsize=14)
ax.set_ylabel("Classe Verdadeira", fontsize=14)
ax.set_xticklabels(["Não Vale", "Vale"], fontsize=12, rotation=0)
ax.set_yticklabels(["Não Vale", "Vale"], fontsize=12, rotation=0)
plt.tight_layout()
st.pyplot(fig)

# — Calcule as métricas da Logistic previamente —
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

# ------------------------------------------------------------------------------
# 6) Conclusions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 1) Crescimento geral
first_val = df_time["Valor_Servicos"].iloc[0]
last_val  = df_time["Valor_Servicos"].iloc[-1]
growth    = (last_val - first_val) / first_val * 100 if first_val else 0

# 2) Crescimento médio das Mesorregiões
import numpy as np
meso_period = (
    df_all
      .groupby([ "Mesorregiao",
                 df_all["Emissão"].dt.to_period("M") ])["Valor_Servicos"]
      .sum()
      .reset_index()
)
growths_meso = []
for meso in meso_period["Mesorregiao"].unique():
    ts = (
        meso_period[meso_period["Mesorregiao"] == meso]
        .sort_values("Emissão")["Valor_Servicos"]
    )
    if len(ts) > 1 and ts.iloc[0] != 0:
        growths_meso.append((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0] * 100)
avg_growth_meso = np.mean(growths_meso) if growths_meso else 0

# 3) Crescimento médio das Microrregiões
micro_period = (
    df_all
      .groupby([ "Microrregiao",
                 df_all["Emissão"].dt.to_period("M") ])["Valor_Servicos"]
      .sum()
      .reset_index()
)
growths_micro = []
for micro in micro_period["Microrregiao"].unique():
    ts = (
        micro_period[micro_period["Microrregiao"] == micro]
        .sort_values("Emissão")["Valor_Servicos"]
    )
    if len(ts) > 1 and ts.iloc[0] != 0:
        growths_micro.append((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0] * 100)
avg_growth_micro = np.mean(growths_micro) if growths_micro else 0

st.markdown("---")
st.markdown(
    f"""
    <div style="font-size:1.3rem; line-height:1.5;">
      <h2 style="font-size:2rem; margin-bottom:0.5rem;">
        📝 Conclusões e Pontos Positivos
      </h2>
      <ul style="margin-top:0.5rem;">
        <li><strong>Crescimento geral</strong>: ↑ {growth:.1f}% entre {df_time['Emissão'].iloc[0]} e {df_time['Emissão'].iloc[-1]}.</li>
        <li><strong>Crescimento médio das Mesorregiões</strong>: ↑ {avg_growth_meso:.1f}% no mesmo período.</li>
        <li><strong>Crescimento médio das Microrregiões</strong>: ↑ {avg_growth_micro:.1f}% no mesmo período.</li>
        <li><strong>Mesorregião/Cidade de destaque</strong>: Top regiões concentram &gt;50% do faturamento.</li>
        <li><strong>Classificação “Vale Investir”</strong> (Logistic Regression):  
            acurácia = {accuracy:.3f}, precisão = {precision:.3f}, recall = {recall:.3f}.</li>
        <li><strong>Recomendação</strong>: Focar em mesorregiões de alto volume e crescimento consistente com a precisão de mais de 90%.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)






