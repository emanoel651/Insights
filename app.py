import streamlit as st
from PIL import Image
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Home", layout="wide")


# ------------------------------------------------------------------------------
# Path até o logo
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
logo_path = ROOT / "meu_projeto\logo.png"

# ------------------------------------------------------------------------------
# Layout: imagem + título
# ------------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 10])  # imagem fica menor, título maior
with col_logo:
    if logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA")
        # remove fundo branco
        pixels = logo.getdata()
        new_pixels = [
            (255, 255, 255, 0) if (r > 240 and g > 240 and b > 240) else (r, g, b, a)
            for r, g, b, a in pixels
        ]
        logo.putdata(new_pixels)
        # define width em pixels ao invés de use_column_width
        st.image(logo, width=150)
    else:
        st.error(f"Logo não encontrado em:\n{logo_path}")

with col_title:
    st.markdown(
        """
        <h1 style="font-size:4rem; margin-top:0; line-height:1.1;">
            📊 Relatório Comercial Instituto Plenum Brasil
        </h1>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ------------------------------------------------------------------------------
# Texto de boas‑vindas (maior)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div style="font-size:1.5rem; line-height:1.6;">
        Bem‑vindo ao <em>Dashboard</em> de Relatórios Comerciais!<br><br>
        Aqui você acessa as vendas dos últimos 18 meses.<br><br>
        <strong>Como usar</strong><br>
        - Abra o menu no canto superior esquerdo (ícone ☰) e escolha:<br>
          &bull; Plenum*<br>
          &bull; Instituto<br>
          &bull; Insights<br><br>
        Cada página carrega métricas, gráficos e tabelas automaticamente.<br><br>
        <strong>LEMBRANDO QUE ALGUNS DADOS SE PERDERAM POR FALTA DE INFORMAÇÃO.</strong><br><br>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
