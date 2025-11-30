import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du risque de CHD",
    page_icon="🫀",
    layout="wide"
)

# CSS personnalisé pour moderniser l'interface
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>🫀 Système de Prédiction du Risque Cardiaque</h1>
        <p>Analyse prédictive basée sur l'intelligence artificielle</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar pour les informations
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
    st.title("ℹ️ À propos")
    st.markdown("""
    ### 🔬 Technologie
    - **Développement** : VS Code
    - **Déploiement** : Streamlit
    - **Modèle** : ML Pipeline
    - **Algorithme** : Régression Logistique + ACP
    
    ### 📊 Dataset
    Source : CHD.csv
    
    ### ⚠️ Avertissement
    Cette application est à but **pédagogique uniquement** et ne remplace en aucun cas un diagnostic médical professionnel.
    """)

# Fonctions de chargement
def clean_categorical(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()  
            .str.lower()  
        )
    return df

@st.cache_resource
def load_model():
    model = joblib.load("Model.pkl")
    return model

model = load_model()

# Section de saisie avec design amélioré
st.markdown("## 📋 Informations du Patient")

with st.form("chd_form"):
    # Trois colonnes pour une meilleure organisation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Données Démographiques")
        age = st.slider("Âge", min_value=10, max_value=100, value=50, help="Âge du patient en années")
        famhist = st.selectbox("Antécédents familiaux", ["Present", "Absent"], 
                               help="Présence de maladies cardiaques dans la famille")
    
    with col2:
        st.markdown("### 🩸 Mesures Cardiovasculaires")
        sbp = st.number_input("Pression systolique (mmHg)", 
                              min_value=80.0, max_value=250.0, value=140.0,
                              help="Pression artérielle systolique")
        ldl = st.number_input("LDL (mmol/L)", 
                              min_value=0.0, max_value=10.0, value=4.0,
                              help="Cholestérol LDL (mauvais cholestérol)")
    
    with col3:
        st.markdown("### 📏 Mesures Corporelles")
        adiposity = st.number_input("Adiposité", 
                                    min_value=0.0, max_value=60.0, value=25.0,
                                    help="Pourcentage de graisse corporelle")
        obesity = st.number_input("Obésité (IMC)", 
                                  min_value=0.0, max_value=60.0, value=30.0,
                                  help="Indice de masse corporelle")
    
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Analyser le Risque Cardiaque")

# Prédiction et affichage des résultats
if submitted:
    input_data = {
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Affichage des données saisies dans un tableau stylisé
    st.markdown("## 📊 Récapitulatif des Données")
    col_display1, col_display2 = st.columns(2)
    
    with col_display1:
        st.metric("Âge", f"{age} ans")
        st.metric("Pression systolique", f"{sbp} mmHg")
        st.metric("LDL", f"{ldl} mmol/L")
    
    with col_display2:
        st.metric("Adiposité", f"{adiposity}")
        st.metric("Obésité (IMC)", f"{obesity}")
        st.metric("Antécédents familiaux", famhist)
    
    # Prédiction
    proba_chd = model.predict_proba(input_df)[0, 1]
    pred_chd = model.predict(input_df)[0]
    
    st.markdown("---")
    st.markdown("## 🎯 Résultats de l'Analyse")
    
    # Affichage du résultat avec jauge visuelle
    col_result1, col_result2 = st.columns([2, 1])
    
    with col_result1:
        # Barre de progression pour la probabilité
        st.markdown("### 📈 Probabilité de Risque CHD")
        st.progress(proba_chd)
        st.markdown(f"<h2 style='text-align: center; color: {'#d32f2f' if proba_chd > 0.5 else '#388e3c'};'>{proba_chd:.1%}</h2>", 
                    unsafe_allow_html=True)
    
    with col_result2:
        st.markdown("### 🏥 Diagnostic")
        if pred_chd == 1:
            st.error("⚠️ **RISQUE ÉLEVÉ**")
            st.markdown("Le modèle détecte un risque important de maladie cardiaque.")
        else:
            st.success("✅ **RISQUE FAIBLE**")
            st.markdown("Le modèle indique un risque réduit de maladie cardiaque.")
    
    # Recommandations
    st.markdown("---")
    st.info("💡 **Recommandation** : Consultez un professionnel de santé pour une évaluation complète et un suivi personnalisé.")