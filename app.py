import streamlit as st

# -------------------------------------------------------------------
# Configuração da página
# -------------------------------------------------------------------
st.set_page_config(page_title="Home", layout="wide")

# -------------------------------------------------------------------
# Título principal
# -------------------------------------------------------------------
st.markdown(
    """
    <h1 style="font-size:4rem; margin-top:0; line-height:1.1;">
        📊 Relatório Comercial Instituto Plenum Brasil
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------------------
# Texto de boas‑vindas (maior)
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size:1.5rem; line-height:1.6;">
        Bem‑vindo ao <em>Dashboard</em> de Relatórios Comerciais!<br><br>
        Aqui você acessa as vendas dos últimos 18 meses.<br><br>
        <strong>Como usar</strong><br>
        - Abra o menu no canto superior esquerdo (ícone ☰) e escolha:<br>
          &bull; Plenum<br>
          &bull; Instituto<br>
          &bull; Insights<br><br>
        Cada página carrega métricas, gráficos e tabelas automaticamente.<br><br>
        <strong>LEMBRANDO QUE ALGUNS DADOS SE PERDERAM POR FALTA DE INFORMAÇÃO.</strong><br><br>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
