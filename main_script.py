# --------------------------- IMPORTS ---------------------------

import streamlit as st  # Librairie principale pour créer l'interface web
from RetrieveFunction import get_mab_info_from_chroma_no_temp  
from langchain.vectorstores import Chroma  # Pour interagir avec la base vectorielle Chroma
from langchain_ollama import OllamaLLM  # Pour utiliser un LLM local via Ollama
from langchain.embeddings import HuggingFaceEmbeddings  # Pour générer les embeddings des textes

# --------------------------- CONFIGURATION DE LA PAGE ---------------------------

# Configuration globale de la page : titre + largeur
st.set_page_config(page_title="Monoclonal Antibody Explorer", layout="wide")

# CSS personnalisé pour styliser toute l’interface CSS injecté dans un bloc markdown
custom_css = """
<style>
body { 
    background-color: #f3f4f6;  /* Couleur de fond douce */
    font-family: 'Segoe UI', sans-serif;  /* Police lisible */
}
h1 {
    color: #4B0082;  /* Violet foncé pour le titre */
    text-align: center;
}
h3, .stMarkdown h2 {
    color: #800080;  /* Violet pour les sous-titres */
    margin-top: 30px;
}
.result-box {
    background-color: #ffffff;  /* Fond blanc pour chaque résultat */
    border: 1px solid #4B0082;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);  /* Légère ombre */
}
strong {
    color: #2c2c2c;  /* Texte en gras plus sombre */
}
input[type="text"] {
    border: 2px solid #4B0082 !important;
    border-radius: 8px !important;
}
button[kind="primary"] {
    background-color: #6a0dad;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------- CHARGEMENT DES RESSOURCES ---------------------------

# Fonction qui charge la base vectorielle Chroma avec les bons embeddings
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")  # Modèle d'embeddings
    return Chroma(persist_directory="./ChromaDB/", embedding_function=embedding_model)  # Chargement de la base Chroma

# Fonction qui charge le LLM local via Ollama
@st.cache_resource
def load_llm():
    return OllamaLLM(model="phi4:latest")  

# Chargement effectif des ressources
vectorstore = load_vectorstore()
llm = load_llm()

# --------------------------- SESSION STATE POUR STOCKER LES RÉSULTATS ---------------------------

# Initialisation des résultats si ce n'est pas encore fait
if "results" not in st.session_state:
    st.session_state.results = None

# --------------------------- INTERFACE UTILISATEUR ---------------------------

# Titre principal de l'application
st.title("🔬 Monoclonal Antibody Explorer")

# Message d'accueil et contexte
st.markdown("""
Welcome to the **Monoclonal Antibody Explorer** web interface.

This tool allows you to extract key biomedical information (targets, effects, mechanisms, domains...) 
from full-text scientific articles indexed in **PubMed Central (PMC)** using semantic search and LLM-powered extraction.

This service is powered by:
- 📚 vector search (ChromaDB)
- 🧠 local LLMs via Ollama
- 🧬 biomedical research articles

**IMGT®**, the international ImMunoGeneTics information system®
""")

# Champ de saisie texte pour le nom du mAb (utilisateur)
mab_name = st.text_input("🧪 Enter monoclonal antibody name (e.g. ansuvimab)", "")

# Si l'utilisateur clique sur le bouton et que le champ n’est pas vide
if st.button("Start Extraction") and mab_name.strip():
    with st.spinner("🔎 Extracting information, please wait..."):
        st.session_state.results = get_mab_info_from_chroma_no_temp(mab_name, vectorstore, llm, k=10)  # Stockage dans la session

    st.success("✅ Information successfully extracted!")

# --------------------------- AFFICHAGE DES RÉSULTATS ---------------------------

# Si des résultats sont disponibles
if st.session_state.results:
    results = st.session_state.results  # Récupération depuis la session

    # 🧠 Synthèse globale sur tous les articles
    st.subheader("🧠 Global Summary")
    st.markdown(
        f"""<div class='result-box'>
        {results["global_description"]}
        </div>""",
        unsafe_allow_html=True
    )

    # 🧾 Section des descriptions
    if st.checkbox("🧾 Show Antibody Descriptions"):
        st.subheader("🧾 Antibody Descriptions")
        for res in results["descriptions"]:
            st.markdown(
                f"""<div class='result-box'>
                <strong>Source article : {res['pmc_id']}</strong><br>→ {res['description']}
                </div>""",
                unsafe_allow_html=True
            )

    # 🎯 Section des cibles biologiques
    if st.checkbox("🎯 Show Biological Targets"):
        st.subheader("🎯 Biological Targets")
        for res in results["targets"]:
            st.markdown(
                f"""<div class='result-box'>
                <strong>Source article : {res['pmc_id']}</strong><br>→ {res['target']}
                </div>""",
                unsafe_allow_html=True
            )

    # ⚙️ Section des mécanismes d'action
    if st.checkbox("⚙️ Show Mechanisms of Action"):
        st.subheader("⚙️ Mechanisms of Action")
        for res in results["mechanisms"]:
            st.markdown(
                f"""<div class='result-box'>
                <strong>Source article : {res['pmc_id']}</strong><br>→ {res['mechanism']}
                </div>""",
                unsafe_allow_html=True
            )

    # 💊 Section des effets thérapeutiques
    if st.checkbox("💊 Show Therapeutic Effects"):
        st.subheader("💊 Therapeutic Effects")
        for res in results["effects"]:
            st.markdown(
                f"""<div class='result-box'>
                <strong>Source article : {res['pmc_id']}</strong><br>→ {res['effect']}
                </div>""",
                unsafe_allow_html=True
            )

    # 🏷️ Section des domaines scientifiques
    if st.checkbox("🏷️ Show Scientific Domains"):
        st.subheader("🏷️ Scientific Domains")
        for res in results["domains"]:
            st.markdown(
                f"""<div class='result-box'>
                <strong>Source article : {res['pmc_id']}</strong><br>→ {res['domain']}
                </div>""",
                unsafe_allow_html=True
            )
# Pied de page 
st.markdown(
    """
    <style>
    .footer {
        color: #666666;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 50px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    <div class="footer">
        &copy; IMGT | IGH | CNRS 2025 - All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
