import streamlit as st
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ModelController import ModelController, text_preprocess

st.set_page_config(
    page_title="Clasificador de texto ODS",
    layout="centered",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(150deg, #f0f4ff 0%, #faf7ff 50%, #f4f0ff 100%);
    min-height: 100vh;
}

/* ── Main container ── */
.block-container {
    max-width: 740px;
    padding-top: 3.5rem;
    padding-bottom: 3rem;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    margin-bottom: 2.8rem;
}
.hero-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 18px #7c3aedaa);
}
.hero h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #1e1b4b;
    letter-spacing: -0.5px;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}

/* ── Card wrapper ── */
.card {
    background: #ffffff;
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    box-shadow: 0 4px 32px rgba(109,40,217,0.08), 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Label encima de textarea (Tu texto) ── */
.input-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.40rem;
}

/* ── Streamlit textarea override ── */
textarea {
    background: #f8f7ff !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.93rem !important;
    caret-color: #7c3aed !important;
    caret-shape: bar !important;
    animation: blink-caret 1.1s step-end infinite;
    resize: vertical;
    transition: border-color 0.2s;
}
textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important;
}

/* ── Boton clasificar ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
    margin-top: 0.9rem;
}
div.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
div.stButton > button:active {
    transform: translateY(0px);
    opacity: 1;
}

/* ── Result box ── */
.result-box {
    margin-top: 1.8rem;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(135deg, #ede9fe 0%, #e0e7ff 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 12px;
    animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #4f46e5;
    margin-bottom: 0.35rem;
}
.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #1e1b4b;
    line-height: 1.2;
}
.result-sub {
    margin-top: 0.3rem;
    font-size: 0.82rem;
    color: #6b7280;
    font-weight: 300;
}

/* ── Warning / error ── */
.stAlert {
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 2.6rem;
    font-size: 0.75rem;
    color: #9ca3af;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

ODS_LABELS = {
    1:  "ODS 1 — Fin de la Pobreza",
    2:  "ODS 2 — Hambre Cero",
    3:  "ODS 3 — Salud y Bienestar",
    4:  "ODS 4 — Educación de Calidad",
    5:  "ODS 5 — Igualdad de Género",
    6:  "ODS 6 — Agua Limpia y Saneamiento",
    7:  "ODS 7 — Energía Asequible y No Contaminante",
    8:  "ODS 8 — Trabajo Decente y Crecimiento Económico",
    9:  "ODS 9 — Industria, Innovación e Infraestructura",
    10: "ODS 10 — Reducción de las Desigualdades",
    11: "ODS 11 — Ciudades y Comunidades Sostenibles",
    12: "ODS 12 — Producción y Consumo Responsables",
    13: "ODS 13 — Acción por el Clima",
    14: "ODS 14 — Vida Submarina",
    15: "ODS 15 — Vida de Ecosistemas Terrestres",
    16: "ODS 16 — Paz, Justicia e Instituciones Sólidas",
    17: "ODS 17 — Alianzas para Lograr los Objetivos",
}

# Carga del modelo (cache habilidato)
@st.cache_resource(show_spinner=False)
def load_model():
    """Instancia ModelController (carga el modelo y lo guarda en caché)."""
    try:
        return ModelController(), None
    except Exception as e:
        return None, str(e)

# Layout
st.markdown("""
<div class="hero">
  <h1>Clasificador de texto ODS</h1>
  <p>Entra el texto que quieres clasificar</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-label">Tu texto</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="",
    placeholder="e.g. «El acceso al agua potable es fundamental para reducir la pobreza extrema en zonas rurales…»",
    height=160,
    label_visibility="collapsed",
)

classify_clicked = st.button("Clasificar")

if classify_clicked:
    if not user_text.strip():
        st.warning("Por favor, ingresa un texto para clasificar.")
    else:
        controller, error = load_model()
        if error:
            st.error(f"Error al cargar el modelo: {error}")
        else:
            with st.spinner("Clasificando texto..."):
                try:
                    prediction = controller.predict([user_text.strip()])[0]
                    label_str = ODS_LABELS.get(
                        int(prediction),
                        f"ODS {prediction}"
                    )
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Clase predicha</div>
                        <div class="result-value">{label_str}</div>
                        <div class="result-sub">Clase: <strong>{prediction}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Ha ocurrido un error al clasificar el texto: {e}")

st.markdown('</div>', unsafe_allow_html=True)