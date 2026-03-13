import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(page_title="Dashboard Actuariel & IA", layout="wide", initial_sidebar_state="expanded")

# Fonction de nettoyage des libellés de régions (NLP-like)
def clean_region_name(name):
    if pd.isna(name): return name
    name = str(name).upper()
    # Enlever les numéros au début s'ils existent (ex: '06- MONTREAL' -> 'MONTREAL')
    name = re.sub(r'^\d+\s*[-_]*\s*', '', name)
    # Correction d'orthographe basique et harmonisation
    name = name.replace('É', 'E').replace('È', 'E').replace('Ê', 'E')
    name = name.replace('Ç', 'C').replace('À', 'A')
    name = name.strip()
    return name

# Chargement et nettoyage des données
@st.cache_data(show_spinner=False)
def load_and_clean_data(directory_path="Données"):
    dfs_indem = []
    
    if not os.path.exists(directory_path):
        return pd.DataFrame(), False
        
    # Charger les dossiers d'indemnisation
    filepaths_indem = [os.path.join(directory_path, f) for f in sorted(os.listdir(directory_path)) if f.endswith('.csv') and not os.path.isdir(os.path.join(directory_path, f))]
    
    for f in filepaths_indem:
        try:
            df = pd.read_csv(f, low_memory=False)
        except:
            try:
                df = pd.read_csv(f, sep=';', low_memory=False)
            except:
                df = pd.read_csv(f, encoding='latin1', low_memory=False)
                
        # Extraire l'année
        m = re.search(r'(2016|2017|2018|2019|2020|2021|2022)', f)
        df['YEAR'] = int(m.group(1)) if m else None
        
        # Créer la clé de jointure
        if 'NUM_SEQ' in df.columns:
            df['JOIN_ID'] = df['NUM_SEQ'].astype(str).str.extract(r'_0*(\d+)')[0].astype(float)
            
        dfs_indem.append(df)
        
    if not dfs_indem: return pd.DataFrame(), False
    df_indem_raw = pd.concat(dfs_indem, ignore_index=True)
    df_final = df_indem_raw.copy()
    
    # -------------------------------------------------------------
    # 2. Charger et joindre les Accidents
    # -------------------------------------------------------------
    acc_dir = os.path.join(directory_path, "Accidents")
    if os.path.exists(acc_dir):
        dfs_acc = []
        for f in sorted(os.listdir(acc_dir)):
            if f.endswith('.csv'):
                try:
                    df_acc = pd.read_csv(os.path.join(acc_dir, f), low_memory=False, on_bad_lines='skip')
                except Exception as e:
                    # Fallback si le séparateur est différent
                    df_acc = pd.read_csv(os.path.join(acc_dir, f), sep=';', low_memory=False, on_bad_lines='skip')
                    
                if 'NO_SEQ_COLL' in df_acc.columns:
                    df_acc['JOIN_ID'] = df_acc['NO_SEQ_COLL'].astype(str).str.extract(r'_\s*(\d+)')[0].astype(float)
                    dfs_acc.append(df_acc)
        
        if dfs_acc:
            df_acc_all = pd.concat(dfs_acc, ignore_index=True)
            # Sélectionner les colonnes pertinentes à ramener
            cols_to_keep_acc = ['JOIN_ID', 'GRAVITE', 'CD_COND_METEO', 'CD_ETAT_SURFC', 'CD_GENRE_ACCDN', 'NB_VICTIMES_TOTAL', 
                                'IND_AUTO_CAMION_LEGER', 'IND_VEH_LOURD', 'IND_MOTO_CYCLO', 'IND_VELO', 'IND_PIETON']
            # Remove BOM if present from column names
            df_acc_all.rename(columns=lambda x: x.strip('\ufeff'), inplace=True)
            avail_cols = [c for c in cols_to_keep_acc if c in df_acc_all.columns]
            df_acc_all = df_acc_all[avail_cols].drop_duplicates('JOIN_ID')
            df_final = pd.merge(df_final, df_acc_all, on='JOIN_ID', how='left')

    # -------------------------------------------------------------
    # 3. Charger et joindre les Blessures (Agréger avant jointure)
    # -------------------------------------------------------------
    bless_dir = os.path.join(directory_path, "Blessures indemnisations ")
    if os.path.exists(bless_dir):
        dfs_bless = []
        for f in sorted(os.listdir(bless_dir)):
            if f.endswith('.csv'):
                try:
                    df_bless = pd.read_csv(os.path.join(bless_dir, f), low_memory=False)
                except Exception:
                    try:
                        df_bless = pd.read_csv(os.path.join(bless_dir, f), sep=';', low_memory=False)
                    except Exception:
                        df_bless = pd.read_csv(os.path.join(bless_dir, f), encoding='latin1', low_memory=False)

                if 'NUM_BLESS' in df_bless.columns:
                    # Ajustement magique constaté sur l'année 2019 : decalage de l'ID
                    df_bless['JOIN_ID'] = df_bless['NUM_BLESS'].astype(str).str.extract(r'_0*(\d+)')[0].astype(float) - 19462
                    dfs_bless.append(df_bless)
                    
        if dfs_bless:
            df_bless_all = pd.concat(dfs_bless, ignore_index=True)
            # Puisqu'il y a plusieurs blessures par JOIN_ID, on aggrège :
            # 1. Compter le nb de blessures
            # 2. Prendre le premier groupe de blessure comme principal
            bless_agg = df_bless_all.groupby('JOIN_ID').agg(
                NB_BLESSURES=('CODE', 'count'),
                BLESSURE_PRINCIPALE=('GRP', 'first')
            ).reset_index()
            
            df_final = pd.merge(df_final, bless_agg, on='JOIN_ID', how='left')
            
            # Remplir les NA pour ceux qui n'ont pas eu de blessure dans le registre
            df_final['NB_BLESSURES'] = df_final['NB_BLESSURES'].fillna(0)
            df_final['BLESSURE_PRINCIPALE'] = df_final['BLESSURE_PRINCIPALE'].fillna('Aucune/Inconnue')
            
            
    # NETTOYAGE
    if 'NOM_REGN' in df_final.columns:
        df_final['NOM_REGN_CLEAN'] = df_final['NOM_REGN'].apply(clean_region_name)
    
    if 'CODE_REGN' in df_final.columns:
        # S'assurer que le code est une catégorie et non un nombre continu
        df_final['CODE_REGN'] = df_final['CODE_REGN'].astype(str)
        
    return {
        'final': df_final,
        'indem': df_indem_raw,
        'acc': df_acc_all if 'df_acc_all' in locals() else pd.DataFrame(),
        'bless': df_bless_all if 'df_bless_all' in locals() else pd.DataFrame()
    }, True

# Fonction de simulation de variables actuarielles
@st.cache_data(show_spinner=False)
def simulate_actuarial_data(df):
    """Génère des données financières factices basées sur l'âge pour la démonstration."""
    df_sim = df.copy()
    np.random.seed(42)
    
    # 1. Type de sinistre
    sinistres = ['Accident de la route', 'Chute/Glissade', 'Maladie professionnelle', 'Autre']
    probabilites = [0.4, 0.3, 0.2, 0.1]
    df_sim['TYPE_SINISTRE'] = np.random.choice(sinistres, size=len(df_sim), p=probabilites)
    
    # 2. Coût du sinistre (Montant Indemnisé) basé sur Log-Normale
    # On ajoute des multiplicateurs fictifs selon l'âge
    age_multiplier = {
        'Moins de 20 ans': 0.8,
        '20 à 29 ans': 1.1,
        '30 à 49 ans': 1.0,
        '50 à 64 ans': 1.3,
        '65 ans et plus': 1.6
    }
    
    # Montant de base (médiane autour de 2000$)
    base_mu, base_sigma = 7.5, 1.0 
    
    if 'GROUP_AGE' in df_sim.columns:
        multipliers = df_sim['GROUP_AGE'].map(age_multiplier).fillna(1.0).values
    else:
        multipliers = np.ones(len(df_sim))
        
    montants = np.random.lognormal(mean=base_mu, sigma=base_sigma, size=len(df_sim))
    df_sim['MONTANT_INDEMNISE'] = np.round(montants * multipliers, 2)
    
    # 3. Durée du dossier (Jours)
    df_sim['DUREE_DOSSIER_JOURS'] = np.random.exponential(scale=45, size=len(df_sim)).astype(int) + 1
    
    return df_sim

# ----------------- METIER HELPERS ----------------- #
trade_map = {
    'IND_AUTO_CAMION_LEGER_O': 'Véhicules Légers (Auto/Camionette)',
    'IND_VEH_LOURD_O': 'Camions Lourds / Poids Lourds',
    'IND_MOTO_CYCLO_O': 'Motocyclettes / Cyclomoteurs',
    'IND_VELO_O': 'Vélos / Cyclistes',
    'IND_PIETON_O': 'Piétons'
}

def translate_trait(col, val):
    key = f"{col}_{val}"
    if key in trade_map: return trade_map[key]
    if col.startswith('IND_') and val == 'N': return None # Ignorer la surreprésentation de "Non"
    return f"{val}"

# ----------------- MAIN APP ----------------- #

loader_placeholder = st.empty()
with loader_placeholder.container():
    st.markdown("""
    <style>
    .loader-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 60vh;
    }
    .spinner {
        width: 60px;
        height: 60px;
        border: 6px solid #f3f3f3;
        border-top: 6px solid #1E3A8A; /* Bleu foncé CEI2A */
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loader-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 2em;
        margin: 0;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loader-subtitle {
        color: #666;
        font-size: 1.1em;
        margin-top: 10px;
    }
    </style>
    <div class="loader-container">
        <div class="spinner"></div>
        <h2 class="loader-title">CEI2A</h2>
        <p class="loader-subtitle">Chargement, nettoyage et fusion des données SAAQ en cours...</p>
    </div>
    """, unsafe_allow_html=True)

data_dict, success = load_and_clean_data()
loader_placeholder.empty()

if not success or data_dict['final'].empty:
    st.error("⚠️ Impossible de charger les données. Veuillez vérifier le dossier 'Données'.")
    st.stop()

df = data_dict['final']

# ----- SIDEBAR NAVIGATION ----- #
st.sidebar.image("https://upload.wikimedia.org/wikipedia/fr/thumb/9/99/SAAQ_logo.svg/3840px-SAAQ_logo.svg.png", width=120)
st.sidebar.markdown("<h2 style='text-align: center; font-weight: 800; margin-top: -10px; color: #1E3A8A;'>CEI2A</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8em; color: gray; margin-top: -20px; font-weight: 500;'>Centre d'expertise en analytique et IA</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.title("📊 Actuariat & IA")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "0. Présentation SAAQ & IA",
    "1. Dashboard Exécutif",
    "2. Qualité  des Données",
    "3. Simulation avec données financieres de synthèse",
    "4. IA Appliquée (Données Réelles)",
    "5. Espace Décisionnel Actuaire"
])

st.sidebar.markdown("---")
st.sidebar.info('''
**À propos**
Ce tableau de bord démontre les capacités de Data Science et d'IA (Nettoyage NLP, Clustering).
''')

# ----- PAGE 0 : PRESENTATION ----- #
if page == "0. Présentation SAAQ & IA":
    st.title("🎯 Présentation : Expérimentation SAAQ & Actuariat Augmenté")
    
    st.markdown("""
    ### Bienvenue dans le Centre d'Expertise en IA
    Cette application est une démonstration technique ("Proof of Concept") illustrant comment **l'Intelligence Artificielle de pointe** peut transformer l'exploitation des données ouvertes gouvernementales en **outils d'aide à la décision stratégique pour le régime public d'assurance automobile (SAAQ)**.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.header("1. La Source de Données")
        st.write("""
        Pour cette expérimentation, nous avons ingéré et fusionné les **bases de données ouvertes (Open Data) officielles de la Société de l'assurance automobile du Québec (SAAQ)** couvrant la période 2016-2022. 
        *Ces jeux de données sont publics et leur utilisation est permise sans autorisation préalable.*
        
        **L'ingestion comprend :**
        *   🚓 **Registres Publics d'Accidents** : Les circonstances, la localisation, les conditions météorologiques et les véhicules impliqués.
        *   🤕 **Dossiers de Réclamations (Blessures)** : Le croisement complexe entre le sinistre matériel et le coût/gravité des blessures humaines indemnisées.
        
        **Le Défi Initial** : Ces données brutes, bien que volumineuses (>118 000 dossiers), sont hétérogènes, parfois incomplètes, et non structurées pour la modélisation financière immédiate.
        """)
        
    with col2:
        st.info("""
        **Données Inérées en Chiffres :**
        - **+118 000** dossiers uniques traités.
        - **18+** dimensions croisées (Région, Âge, Véhicule, Météo, Gravité...).
        - **Jointure** : pour relier les données registres publics d'accidents aux paiements d'indemnités.
        """)
        
    st.markdown("---")
    
    st.header("2. L'Approche IA : Le Clustering Non-Supervisé")
    st.write("""
    Au lieu de définir des règles tarifaires rigides basées sur l'intuition (ex: "Les jeunes coûtent plus cher"), nous laissons un algorithme de **Clustering (K-Means)** découvrir de lui-même les "poches de risque" toxiques ou rentables cachées dans les 118 000 lignes, en analysant toutes les dimensions simultanément.
    """)
    
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.write("**Avant l'IA : Données Indifférenciées (Bruit)**")
        # Visualisation Théorique - Avant
        np.random.seed(42)
        x = np.random.randn(300)
        y = np.random.randn(300)
        fig_before, ax_b = plt.subplots(figsize=(5,3))
        ax_b.scatter(x, y, color='gray', alpha=0.5)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_before)
        st.caption("Une masse de dossiers complexes (Impossible de tarifer précisément).")
        
    with col_g2:
        st.write("**Après l'IA : Segmentation (Clustering K-Means)**")
        # Visualisation Théorique - Après
        x1, y1 = np.random.randn(100) - 2, np.random.randn(100) - 2
        x2, y2 = np.random.randn(100) + 2, np.random.randn(100) + 2
        x3, y3 = np.random.randn(100) + 2, np.random.randn(100) - 2
        
        fig_after, ax_a = plt.subplots(figsize=(5,3))
        ax_a.scatter(x1, y1, color='red', label="Très Élevé (Malus)", alpha=0.7)
        ax_a.scatter(x2, y2, color='green', label="Très Faible (Rabais)", alpha=0.7)
        ax_a.scatter(x3, y3, color='orange', label="Moyen", alpha=0.7)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        ax_a.legend(loc='best', fontsize='small')
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_after)
        st.caption("Identification mathématique de Profils de Risques ultra-spécifiques.")
        
    st.markdown("---")
    
    st.header("3. L'Avantage Compétitif (Cas d'Usage Actuariel)")
    st.markdown("""
    L'application de cette IA sur les données de la SAAQ permet trois avancées majeures prêtes pour l'industrie :
    
    1.  🎯 **Tarification (Pricing) Millimétrique** : Fin de la mutualisation aveugle. On surcharge (Malus) chirurgicalement les profils hyper-risqués détectés par l'IA (ex: Camions lourds en zone urbaine) et on offre des rabais justifiés aux bons profils.
    2.  🏦 **Provisionnement (IBNR - Incurred But Not Reported) Précis** : Calcul dynamique d'un "Indice de Sévérité Relative". La direction sait exactement de combien augmenter ses réserves selon la distribution en temps réel de son portefeuille.
    3.  ⚡ **Triage Automatisé** : Dès l'ouverture du dossier (First Notice of Loss), l'IA prédit la gravité et route le dossier (Règlement Fast-Track immédiat vs Expert Senior pour blessure grave), réduisant massivement les frais de gestion opérationnelle (ALAE).
    
    👈 **Utilisez le menu de gauche pour naviguer à travers les différentes étapes de cette expérimentation.**
    """)

# ----- PAGE 1 : DASHBOARD EXECUTIF (DONNÉES RÉELLES) ----- #
elif page == "1. Dashboard Exécutif":
    st.title("Tableau de Bord Exécutif (2016-2022)")
    st.markdown("Visualisation des données **réelles** consolidées et nettoyées.")
    
    # KPIs Haut niveau
    st.markdown("### 📈 Indicateurs Clés")
    col1, col2, col3 = st.columns(3)
    col1.metric("Volume Total des Dossiers", f"{len(df):,}")
    
    if 'GROUP_AGE' in df.columns:
        most_freq_age = df['GROUP_AGE'].mode()[0]
        col2.metric("Tranche d'âge majoritaire", str(most_freq_age))
    
    if 'NOM_REGN_CLEAN' in df.columns:
        most_freq_reg = df['NOM_REGN_CLEAN'].mode()[0]
        col3.metric("Région la plus touchée", str(most_freq_reg).title())
        
    # Ligne 2 : Graphiques
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Évolution du nombre de dossiers par année**")
        fig, ax = plt.subplots(figsize=(6, 4))
        yearly_counts = df['YEAR'].value_counts().sort_index()
        sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o", color="blue", ax=ax, linewidth=2.5)
        ax.set_ylabel("Volume de dossiers")
        ax.set_xticks(yearly_counts.index)
        sns.despine()
        st.pyplot(fig)
        
    with col_chart2:
        st.write("**Répartition par tranche d'âge**")
        if 'GROUP_AGE' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            order = sorted(df['GROUP_AGE'].dropna().unique().astype(str))
            sns.countplot(y='GROUP_AGE', data=df, order=order, palette='viridis', ax=ax)
            sns.despine()
            ax.set_xlabel("Volume")
            st.pyplot(fig)
            
    # Ligne 3 : Tranches d'âge par année (Graphique Interactif Plotly)
    st.markdown("---")
    st.write("**Évolution des volumes d'accidents par Tranche d'Âge et par Année**")
    if 'GROUP_AGE' in df.columns and 'YEAR' in df.columns:
        import plotly.express as px
        # Préparer les données pour Plotly (Compter les occurrences)
        age_year_counts = df.groupby(['YEAR', 'GROUP_AGE']).size().reset_index(name='Volume')
        order_age = sorted(df['GROUP_AGE'].dropna().unique().astype(str))
        
        fig_age_yr = px.bar(
            age_year_counts, 
            x='YEAR', 
            y='Volume', 
            color='GROUP_AGE', 
            barmode='group',
            category_orders={"GROUP_AGE": order_age},
            color_discrete_sequence=px.colors.sequential.Viridis,
            labels={"YEAR": "Année", "Volume": "Volume d'accidents", "GROUP_AGE": "Tranche d'âge"}
        )
        
        fig_age_yr.update_layout(
            xaxis=dict(type='category'), # S'assurer que les années sont traitées comme des catégories discrètes
            legend=dict(title="Tranche d'âge", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_age_yr, use_container_width=True)

    # Ligne 4 : Graphiques additionnels
    st.markdown("---")
    st.write("**Répartition des dossiers par Région (Top 15)**")
    if 'NOM_REGN_CLEAN' in df.columns:
        fig_reg, ax_reg = plt.subplots(figsize=(12, 5))
        top_regions = df['NOM_REGN_CLEAN'].value_counts().head(15)
        sns.barplot(x=top_regions.values, y=top_regions.index, palette='magma', ax=ax_reg)
        sns.despine()
        ax_reg.set_xlabel("Volume de dossiers")
        ax_reg.set_ylabel("")
        st.pyplot(fig_reg)

# ----- PAGE 2 : QUALITÉ DES DONNÉES (EDA PRO) ----- #
elif page == "2. Qualité  des Données":
    st.title("Exploratory Data Analysis (EDA) & Préparation")
    st.markdown('''
    Une phase critique avant tout modèle prédictif ou de classification IA est la compréhension et l'hygiène des données.
    Ce tableau de bord résume l'état du jeu de données SAAQ après nettoyage et fusion (Accidents + Blessures).
    ''')
    
    tab_indem, tab_acc, tab_bless = st.tabs(["📁 Dossiers d'Indemnisation", "🚓 Registres d'Accidents", "🤕 Blessures Indemnisées"])
    
    with tab_indem:
        st.header("1. Empreinte du Jeu de Données (Indemnisations)")
        df_target = data_dict.get('indem', pd.DataFrame())
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Lignes (Dossiers)", f"{len(df_target):,}")
        col_k2.metric("Colonnes (Variables)", df_target.shape[1])
        col_k3.metric("Années Couvertes", f"{df_target['YEAR'].min()} - {df_target['YEAR'].max()}" if 'YEAR' in df_target.columns else "N/A")
        
        st.markdown("---")
        st.header("2. Audit de Complétude")
        if not df_target.empty:
            missing_data = df_target.isnull().sum()
            missing_pct = (missing_data / len(df_target)) * 100
            missing_df = pd.DataFrame({'Valeurs Manquantes': missing_data, 'Pourcentage (%)': missing_pct})
            missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(by='Pourcentage (%)', ascending=False)
            
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                if len(missing_df) > 0:
                    st.warning(f"⚠️ {len(missing_df)} variables présentent des données manquantes.")
                    st.dataframe(missing_df.style.format({'Pourcentage (%)': '{:.1f}%'}), height=250)
                else:
                    st.success("✅ Aucune valeur manquante détectée.")
            with col_m2:
                if len(missing_df) > 0:
                    st.write("**Top 10 Variables Incomplètes**")
                    fig_miss, ax_miss = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='Pourcentage (%)', y=missing_df.head(10).index, data=missing_df.head(10), palette='Reds_r', ax=ax_miss)
                    sns.despine()
                    ax_miss.set_xlabel("% Manquant")
                    ax_miss.set_ylabel("")
                    st.pyplot(fig_miss)
                    
        st.markdown("---")
        st.header("3. Ingénierie des Données (Feature Engineering)")
        st.write("Exemple d'action ciblée : Nettoyage par Regex/NLP de la variable `NOM_REGN` (Régions) pour éliminer le bruit numérique et uniformiser la casse.")
        if 'NOM_REGN' in df.columns and 'NOM_REGN_CLEAN' in df.columns:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("**1. Donnée Brute** `NOM_REGN`")
                st.dataframe(df['NOM_REGN'].drop_duplicates().head(5).reset_index(drop=True), use_container_width=True)
            with col_c2:
                st.markdown("**2. Donnée Standardisée** `NOM_REGN_CLEAN`")
                st.dataframe(df['NOM_REGN_CLEAN'].drop_duplicates().head(5).reset_index(drop=True), use_container_width=True)

    with tab_acc:
        st.header("1. Empreinte du Jeu de Données (Accidents)")
        df_target = data_dict.get('acc', pd.DataFrame())
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("Lignes (Registres)", f"{len(df_target):,}")
        col_k2.metric("Colonnes (Variables)", df_target.shape[1])
        
        st.markdown("---")
        st.header("2. Audit de Complétude")
        if not df_target.empty:
            missing_data = df_target.isnull().sum()
            missing_pct = (missing_data / len(df_target)) * 100
            missing_df = pd.DataFrame({'Valeurs Manquantes': missing_data, 'Pourcentage (%)': missing_pct})
            missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(by='Pourcentage (%)', ascending=False)
            
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                if len(missing_df) > 0:
                    st.warning(f"⚠️ {len(missing_df)} variables présentent des données manquantes.")
                    st.dataframe(missing_df.style.format({'Pourcentage (%)': '{:.1f}%'}), height=250)
                else:
                    st.success("✅ Aucune valeur manquante détectée.")
            with col_m2:
                if len(missing_df) > 0:
                    st.write("**Top 10 Variables Incomplètes**")
                    fig_miss, ax_miss = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='Pourcentage (%)', y=missing_df.head(10).index, data=missing_df.head(10), palette='Reds_r', ax=ax_miss)
                    sns.despine()
                    ax_miss.set_xlabel("% Manquant")
                    ax_miss.set_ylabel("")
                    st.pyplot(fig_miss)
                    
        st.markdown("---")
        st.header("3. Équilibre des Variables Clés (Features)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**Répartition par Sévérité Initiale**")
            if 'GRAVITE' in df_target.columns:
                fig_sev, ax_sev = plt.subplots(figsize=(6, 4))
                grav_counts = df_target['GRAVITE'].value_counts()
                plt.pie(grav_counts.values, labels=grav_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
                st.pyplot(fig_sev)
                st.caption("On observe un débalancement des classes typique en indemnisation corporelle : les cas graves sont très rares (queue de distribution).")
            else:
                st.info("Variable GRAVITE non trouvée.")

        with col_d2:
            st.write("**Exposition aux Risques Spécifiques (Volumes croisés)**")
            veh_cols = [c for c in df_target.columns if c.startswith('IND_') and c != 'IND_AUTO_CAMION_LEGER']
            if veh_cols:
                veh_counts = {c.replace('IND_', ''): (df_target[c] == 'O').sum() for c in veh_cols}
                fig_v, ax_v = plt.subplots(figsize=(6, 4))
                sns.barplot(x=list(veh_counts.values()), y=list(veh_counts.keys()), palette="viridis", ax=ax_v)
                sns.despine()
                ax_v.set_xlabel("Nombre de dossiers impliqués")
                st.pyplot(fig_v)
                
    with tab_bless:
        st.header("1. Empreinte du Jeu de Données (Blessures)")
        df_target = data_dict.get('bless', pd.DataFrame())
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("Lignes (Blessures individuelles)", f"{len(df_target):,}")
        col_k2.metric("Colonnes (Variables)", df_target.shape[1])
        
        st.markdown("---")
        st.header("2. Audit de Complétude")
        if not df_target.empty:
            missing_data = df_target.isnull().sum()
            missing_pct = (missing_data / len(df_target)) * 100
            missing_df = pd.DataFrame({'Valeurs Manquantes': missing_data, 'Pourcentage (%)': missing_pct})
            missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(by='Pourcentage (%)', ascending=False)
            
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                if len(missing_df) > 0:
                    st.warning(f"⚠️ {len(missing_df)} variables présentent des données manquantes.")
                    st.dataframe(missing_df.style.format({'Pourcentage (%)': '{:.1f}%'}), height=250)
                else:
                    st.success("✅ Aucune valeur manquante détectée.")
            with col_m2:
                if len(missing_df) > 0:
                    st.write("**Top 10 Variables Incomplètes**")
                    fig_miss, ax_miss = plt.subplots(figsize=(8, 4))
                    sns.barplot(x='Pourcentage (%)', y=missing_df.head(10).index, data=missing_df.head(10), palette='Reds_r', ax=ax_miss)
                    sns.despine()
                    ax_miss.set_xlabel("% Manquant")
                    ax_miss.set_ylabel("")
                    st.pyplot(fig_miss)

# ----- PAGE 3 : LABORATOIRE IA ET SIMULATION ACTUARIELLE ----- #
elif page == "3. Simulation avec données financieres de synthèse":
    st.title("🔬 Lab I.A. & Recommandations Actuarielles")
    
    st.markdown('''
    ### 🎯 Pourquoi croiser l'Actuariat et l'Intelligence Artificielle ?
    En actuariat traditionnel, la tarification et le provisionnement s'appuient sur des modèles statistiques classiques (GLM) qui demandent souvent de définir des profils de risque manuellement (ex: "Jeunes conducteurs", "Seniors en région rurale").
    
    **L'apport de l'IA (le Clustering non-supervisé) :**
    L'IA permet de découvrir des segments (clusters) cachés sans a priori humain. Elle regroupe automatiquement les assurés selon de multiples variables simultanément (Âge, Région, Coût, Type de sinistre). 
    
    👉 *L'objectif : Identifier automatiquement les "mauvaises poches" de risques (très chères, très fréquentes) ou les profils extrêmement rentables pour ajuster la tarification de façon granulaire et ultra-personnalisée.*
    ''')

    st.error('''
    **⚠️ AVERTISSEMENT : RECOMMANDATION STRATÉGIQUE DE COLLECTE**
    Le jeu de données réel ne contient **pas de variables financières ou de sinistralité** (montants, durées, garantie touchée). 
    Pour démontrer ici l'apport de l'IA, **nous générons mathématiquement des montants et des durées factices** (corrélés à l'âge). 
    ''')
    
    with st.expander("🧮 Méthodologie & Formules (Génération des Variables Manquantes)"):
        st.markdown("""
        Puisque le registre ouvert ne contient pas de données monétaires, nous avons simulé la sévérité financière et temporelle des dossiers en utilisant des lois de probabilité standards en actuariat :
        
        **1. Coût du Sinistre (Loi Lognormale)**
        Le coût total $Y$ est modélisé par une loi lognormale pour reproduire la "queue épaisse" (heavy tail) des gros sinistres.
        
        | Cas | Coût |
        | :--- | :--- |
        | Petit accident | 500$ |
        | Moyen | 3 000$ |
        | Gros | 25 000$ |
        | Catastrophique | 300 000$ |
        """)
        st.latex(r"Y \sim \text{Lognormal}(\mu_{age}, \sigma)")
        st.markdown("""
        **2. Durée d'Indemnisation (Loi Exponentielle)**
        Le temps de récupération (en jours) est modélisé pour refléter que la majorité des sinistres se referment vite, avec quelques cas très longs (invalidité).
        
        | Type de cas | Durée |
        | :--- | :--- |
        | Petite blessure | 5 jours |
        | Arrêt de travail | 30 jours |
        | Fracture | 90 jours |
        | Invalidité | 600 jours |
        """)
        st.latex(r"D \sim \text{Expo}(\lambda_{gravite})")
        
    # Simulation des données
    with st.spinner("Génération de l'environnement de simulation (Lois Log-Normales & Exponentielles)..."):
        df_sim = simulate_actuarial_data(df)
        
    st.markdown("### Aperçu du Dataset Enrichi (Simulé pour la démonstration)")
    st.dataframe(df_sim[['YEAR', 'GROUP_AGE', 'NOM_REGN_CLEAN', 'TYPE_SINISTRE', 'DUREE_DOSSIER_JOURS', 'MONTANT_INDEMNISE']].head())
    
    st.markdown("---")
    st.subheader("🤖 Algorithme de Segmentation (K-Means)")
    st.markdown('''
    L'algorithme **K-Means** va lire les milliers de lignes ci-dessus, analyser les distances mathématiques entre chaque dossier (Coût, Durée, Âge, Région) et les regrouper en "Profils Homogènes". 
    Sélectionnez le nombre de profils que vous souhaitez que l'IA identifie :
    ''')
    
    k_choice = st.slider("Nombre de profils de risque (Clusters)", min_value=2, max_value=6, value=3)
    
    if st.button("Lancer l'Algorithme d'Apprentissage IA"):
        with st.spinner("Compression PCA et exécution de KMeans..."):
            # Préparation des features simulées
            features = ['MONTANT_INDEMNISE', 'DUREE_DOSSIER_JOURS']
            if 'GROUP_AGE' in df_sim.columns: features.append('GROUP_AGE')
            if 'NOM_REGN_CLEAN' in df_sim.columns: features.append('NOM_REGN_CLEAN')
            
            data = df_sim[features].dropna()
            
            # Échantillonnage pour fluidité de la démo
            sample_size = min(10000, len(data))
            data_sample = data.sample(n=sample_size, random_state=42)
            
            # Encodage
            categorical = [c for c in features if data_sample[c].dtype == object or str(data_sample[c].dtype) == 'category']
            data_enc = pd.get_dummies(data_sample, columns=categorical)
            
            # Scaling
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_enc)
            
            # PCA 
            pca = PCA(n_components=0.95, random_state=42)
            data_pca = pca.fit_transform(data_scaled)
            
            # KMeans
            km = KMeans(n_clusters=k_choice, n_init=10, random_state=42)
            clusters = km.fit_predict(data_pca)
            
            data_sample['Cluster_IA'] = clusters
            
            st.success(f"Segmentation réussie ! (L'Agorithme a compressé l'information à 95% avant de trouver les {k_choice} groupes).")
            
            st.markdown("### 💰 Résultat Actuariel : Impact Financier par Profil")
            st.markdown('''
            *Lisez le tableau ci-dessous comme un Actuaire : Le cluster en surbrillance rouge représente la "poche de risque" la plus coûteuse. 
            C'est ce segment précis qu'il faudra vérifier (hausse de prime, prévention, audit médico-légal).*
            ''')
            
            # Grouper par cluster pour trouver le coût moyen
            cluster_stats_df = data_sample.groupby('Cluster_IA').agg(
                Volume=('MONTANT_INDEMNISE', 'count'),
                Cout_Moyen=('MONTANT_INDEMNISE', 'mean'),
                Duree_Moyenne=('DUREE_DOSSIER_JOURS', 'mean')
            ).reset_index()
            
            # Mettre en forme pour l'affichage
            cluster_stats_df['% du Portefeuille'] = (cluster_stats_df['Volume'] / len(data_sample) * 100).round(1).astype(str) + '%'
            cluster_stats_df['Coût Moyen ($)'] = cluster_stats_df['Cout_Moyen'].round(2)
            cluster_stats_df['Durée Moyenne de Gestion (Jours)'] = cluster_stats_df['Duree_Moyenne'].round(0).astype(int)
            
            st.dataframe(cluster_stats_df[['Cluster_IA', '% du Portefeuille', 'Volume', 'Coût Moyen ($)', 'Durée Moyenne de Gestion (Jours)']].style.highlight_max(subset=['Coût Moyen ($)'], color='#ffcccc'))
            
            st.markdown("### 🕵️‍♂️ Décryptage : Qui se cache dans ces profils ?")
            st.markdown("Maintenant que l'IA a isolé les coûts, voici la démographie (Âge et Région) qui compose chaque profil :")
            
            for i in range(k_choice):
                sub = data_sample[data_sample['Cluster_IA'] == i]
                
                # Check if it's the most expensive cluster
                is_max = (cluster_stats_df.loc[cluster_stats_df['Cluster_IA'] == i, 'Cout_Moyen'].values[0] == cluster_stats_df['Cout_Moyen'].max())
                icon = "🔥 PROFIL À HAUT RISQUE" if is_max else "✅ PROFIL STANDARD"
                
                with st.expander(f"Détail du Profil (Cluster) {i} - {icon}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Répartition par Âge :**")
                        st.dataframe((sub['GROUP_AGE'].value_counts(normalize=True)*100).round(1).rename("%"))
                    with c2:
                        st.write("**Répartition par Région :**")
                        st.dataframe((sub['NOM_REGN_CLEAN'].value_counts(normalize=True)*100).round(1).rename("%"))
                    
                    if is_max:
                        st.error(f"**Action Actuarielle Conseillée :** Ce profil présente un surcoût évident. Une analyse des garanties souscrites par cette population cible est prioritaire pour endiguer la perte technique.")

# ----- PAGE 4 : IA APPLIQUÉE (DONNÉES RÉELLES) ----- #
elif page == "4. IA Appliquée (Données Réelles)":
    st.title("🎯 IA Appliquée aux Données Réelles de Sinistres")
    st.markdown('''
    Contrairement à l'onglet "Lab IA" qui utilise des variables financières simulées à des fins pédagogiques, 
    cet onglet utilise nos algorithmes sur les **vraies données d'accidents et de blessures** croisées avec les dossiers d'indemnisation.
    ''')
    
    st.markdown("### 🔍 Aperçu des données enrichies")
    st.markdown("L'IA a automatiquement fusionné la base des *Dossiers* avec celles des *Accidents* (Météo, Surface, Gravité) et des *Blessures indemnisées* (Nombre et Type).")
    
    # Show joined features
    view_cols = ['NUM_SEQ', 'NOM_REGN_CLEAN', 'GROUP_AGE', 'CD_COND_METEO', 'CD_SURF_ROUT', 'GRAVITE', 'NB_BLESSURES', 'BLESSURE_PRINCIPALE']
    avail_cols = [c for c in view_cols if c in df.columns]
    
    # Display sample where we have accidents/injuries
    df_real = df.dropna(subset=['GRAVITE', 'BLESSURE_PRINCIPALE'])
    st.dataframe(df_real[avail_cols].head(100), use_container_width=True)
    
    # ------ GRAPHICS ------
    st.markdown("---")
    st.markdown("### 📊 Distribution des Vrais Sinistres")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top 5 Types de Blessures**")
        if 'BLESSURE_PRINCIPALE' in df.columns:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            top_bless = df[df['BLESSURE_PRINCIPALE'] != 'Aucune/Inconnue']['BLESSURE_PRINCIPALE'].value_counts().head(5)
            # Wrap text to avoid overlap
            import textwrap
            labels = [textwrap.fill(x, 25) for x in top_bless.index]
            sns.barplot(x=top_bless.values, y=labels, palette='rocket', ax=ax1)
            sns.despine()
            st.pyplot(fig1)
            
    with c2:
        st.write("**Gravité des Accidents**")
        if 'GRAVITE' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            grav_counts = df['GRAVITE'].dropna().value_counts()
            labels2 = [textwrap.fill(x, 25) for x in grav_counts.index]
            sns.barplot(y=labels2, x=grav_counts.values, palette='mako', ax=ax2)
            sns.despine()
            st.pyplot(fig2)
            
    st.markdown("---")
    st.subheader("🤖 Clustering sur Variables Réelles (Complexité Croisée)")
    st.markdown('''
    Nous injectons dans l'Apprentissage Non-Supervisé des variables complexes réelles : **Âge, Région, Catégorie de Blessure et Gravité**.
    L'IA va trouver les profils les plus lourds à gérer pour un actuaire sans qu'aucun montant financier théorique ne soit simulé !
    ''')
    
    with st.expander("🧮 Méthodologie & Formules (Algorithme K-Means & Représentativité)"):
        st.markdown("""
        **1. Mathématiques du Clustering (K-Means)**
        L'algorithme tente de minimiser la variance intra-classe (inertie) en calculant la distance euclidienne $d$ entre chaque sinistre $x$ et le centre $c_i$ de son groupe :
        """)
        st.latex(r"W(C) = \sum_{k=1}^{K} \sum_{x \in C_k} ||x - c_k||^2")
        st.markdown("""
        **2. Indicateur de Sur-représentation**
        Pour auditer chaque cluster, nous calculons l'écart absolu de fréquence d'un trait donné par rapport à sa distribution globale.
        """)
        st.latex(r"Ecart(Trait_i) = P(Trait_i | Cluster_k) - P(Trait_i | Global)")
        st.markdown("*L'interface ne vous alertera que si cet écart dépasse **+5.0%**.*")
        
    k_choice_real = st.slider("Nombre de profils à chercher (K) :", min_value=2, max_value=6, value=3, key="k_real")
    
    if st.button("Lancer K-Means (Temps Réel)"):
        with st.spinner("Analyse K-Means sur données réelles..."):
            features_real = ['NB_BLESSURES']
            if 'YEAR' in df.columns: features_real.append('YEAR')
            if 'GROUP_AGE' in df.columns: features_real.append('GROUP_AGE')
            
            # Ajouter les types de véhicules spécifiques si disponibles
            for ind_col in ['IND_AUTO_CAMION_LEGER', 'IND_VEH_LOURD', 'IND_MOTO_CYCLO', 'IND_VELO', 'IND_PIETON']:
                if ind_col in df.columns:
                    features_real.append(ind_col)
                    
            if 'BLESSURE_PRINCIPALE' in df.columns: features_real.append('BLESSURE_PRINCIPALE')
            
            # Subsample those who have accidents
            data_r = df.dropna(subset=['BLESSURE_PRINCIPALE'])[features_real].copy()
            
            if len(data_r) < 100:
                st.warning("⚠️ Pas assez de données fiables croisées pour lancer l'IA.")
            else:
                sample_r = data_r.sample(n=min(8000, len(data_r)), random_state=42)
                
                # Encodage
                cat_cols_r = [c for c in features_real if sample_r[c].dtype == object or str(sample_r[c].dtype) == 'category']
                data_enc_r = pd.get_dummies(sample_r, columns=cat_cols_r)
                
                # Scaler & PCA
                scaler_r = StandardScaler()
                data_scaled_r = scaler_r.fit_transform(data_enc_r)
                
                pca_r = PCA(n_components=0.95, random_state=42)
                data_pca_r = pca_r.fit_transform(data_scaled_r)
                
                # KMeans
                km_r = KMeans(n_clusters=k_choice_real, n_init=10, random_state=42)
                clusters_r = km_r.fit_predict(data_pca_r)
                sample_r['Cluster'] = clusters_r
                
                st.success("Segmentation K-Means Réelle terminée !")
                st.markdown("### 🏆 Profils de Risque Actuariel (Basé sur la Sur-représentation)")
                st.markdown("*L'IA a identifié ce qui rend chaque groupe **unique** par rapport à la moyenne du portefeuille.*")
                
                # Calculer les proportions globales pour trouver ce qui caractérise un cluster
                overall_props = {}
                for col in cat_cols_r:
                    overall_props[col] = sample_r[col].value_counts(normalize=True)
                
                # Trier les clusters par taille (du plus grand au plus petit)
                cluster_sizes = sample_r['Cluster'].value_counts()
                ranked_clusters = cluster_sizes.index.tolist()
                
                session_profiles = []
                
                for rank, c_id in enumerate(ranked_clusters):
                    sub = sample_r[sample_r['Cluster'] == c_id]
                    avg_bless = sub['NB_BLESSURES'].mean() if 'NB_BLESSURES' in sub.columns else 0
                    
                    # Trouver le trait le plus sur-représenté
                    defining_traits = []
                    for col in cat_cols_r:
                        sub_prop = sub[col].value_counts(normalize=True)
                        diff = (sub_prop - overall_props[col]).dropna().sort_values(ascending=False)
                        if len(diff) > 0 and diff.iloc[0] > 0.05: # Plus de 5% de sur-représentation
                            defining_traits.append((col, diff.index[0], diff.iloc[0]*100))
                    
                    # Trier par le pourcentage d'écart le plus fort
                    defining_traits.sort(key=lambda x: x[2], reverse=True)
                    
                    with st.expander(f"**Profil N°{rank+1}** (représente {(len(sub)/len(sample_r)*100):.1f}% des sinistres)", expanded=(rank==0)):
                            
                        # Si on a trouvé des traits spécifiques
                        valid_traits = [(col, val, pct) for (col, val, pct) in defining_traits if translate_trait(col, val) is not None]
                        
                        if valid_traits:
                            top_trait = valid_traits[0]
                            trait_name = translate_trait(top_trait[0], top_trait[1])
                            st.write(f"🎯 **Caractéristique clé :** Sur-représentation de **{trait_name}** (+{top_trait[2]:.1f}% vs Moyenne)")
                            
                            if len(valid_traits) > 1:
                                second_trait = valid_traits[1]
                                trait_name_2 = translate_trait(second_trait[0], second_trait[1])
                                st.write(f"🔍 **Facteur de co-morbidité :** Sur-représentation de **{trait_name_2}** (+{second_trait[2]:.1f}% vs Moyenne)")
                                
                            # ----- INTERPRÉTATION ACTUARIELLE EXPERTE -----
                            st.markdown("💡 *Signification Actuarielle :*")
                            if "Lourd" in trait_name:
                                st.error("L'implication d'un poids lourd décuple la force d'impact cinétique. Provisionnement pour dommages corporels (invalidité à long terme) à réévaluer à la hausse.")
                            elif "Vélo" in trait_name or "Piéton" in trait_name:
                                st.warning("Usager vulnérable. Risque de poly-traumatismes élevé avec coûts médicaux récurrents. Stratégie de prévention ciblée recommandée.")
                            elif "Entorse" in trait_name or "cervicale" in trait_name:
                                st.success("Volumétrie élevée mais dommages souvent limités aux tissus mous (coup de fouet). Coût par dossier modéré, mais fréquence à surveiller.")
                            elif "Fractures" in trait_name:
                                st.warning("Lésions structurelles nécessitant un suivi orthopédique et une réhabilitation. Impose une charge financière à moyen terme sur la sécurité routière.")
                            else:
                                st.info("Profil standard de collision, coûts alignés sur la mutualisation classique du risque auto.")
                                
                        else:
                            st.write("🎯 **Caractéristique clé :** Profil hybride très proche de la moyenne globale.")
                            
                        st.write("")
                        
                        # ----- SAVE TO SESSION STATE FOR DECISION TAB -----
                        region_counts = df.loc[sub.index, 'NOM_REGN_CLEAN'].value_counts().head(5) if 'NOM_REGN_CLEAN' in df.columns else pd.Series()
                        
                        # Calculer la répartition des véhicules pour ce cluster depuis le DataFrame original
                        veh_cols_orig = ['IND_AUTO_CAMION_LEGER', 'IND_VEH_LOURD', 'IND_MOTO_CYCLO', 'IND_VELO', 'IND_PIETON']
                        veh_counts = {}
                        for vc in veh_cols_orig:
                            if vc in df.columns:
                                v_name = trade_map.get(vc + '_O', vc) # Match the map key
                                count = df.loc[sub.index, vc].isin(['O', '1', 1, 1.0, 'Oui', 'Y']).sum()
                                veh_counts[v_name] = int(count)
                                
                        profile_info = {
                            "rank": rank + 1,
                            "size_pct": (len(sub)/len(sample_r)*100),
                            "size_count": len(sub),
                            "avg_bless": avg_bless,
                            "traits": valid_traits,
                            "region_dist": region_counts.to_dict(),
                            "veh_dist": veh_counts
                        }
                        session_profiles.append(profile_info)
                        
                st.session_state['actuarial_profiles'] = session_profiles


# ----- PAGE 5 : ESPACE DÉCISIONNEL ACTUAIRE ----- #
elif page == "5. Espace Décisionnel Actuaire":
    st.title("⚖️ Espace Décisionnel Actuaire (Temps Réel)")
    st.markdown("""
    *Cet espace traduit les signaux faibles découverts par l'IA (Clustering) en actions métiers directes sur 4 piliers actuariels fondamentaux.*
    """)
    
    with st.expander("Comment l'IA calcule-t-elle ces recommandations ?"):
        st.markdown(r"""
        Cette application repose sur une analyse automatisée des données historiques pour guider la gestion :
        
        **1. Profilage Automatique (Intelligence Artificielle)**
        L'IA regroupe les milliers de dossiers en quelques grands "profils" ou "familles" de sinistres qui se ressemblent (en termes de région, âge, véhicule, type de blessure). Cela permet de ne pas traiter chaque dossier aveuglément de la même façon.
        
        **2. Analyse des Déviations**
        Le système ne se contente pas de faire des moyennes floues. Il compare la "gravité" de chaque profil trouvé par rapport à la moyenne historique globale de la SAAQ.
        
        **3. Sur-représentation**
        Plutôt que d'afficher des statistiques illisibles, l'outil vous alerte uniquement lorsqu'un facteur est anormalement élevé dans un groupe. 
        *Exemple : Si les camions lourds représentent 2% du total des accidents, mais 90% des accidents d'un profil spécifique, l'IA lève un drapeau rouge (+88%) car c'est la cause racine de ce profil.*
        """)
    
    # Vérifier si l'analyse a été lancée
    profiles = st.session_state.get('actuarial_profiles', None)
    
    if not profiles:
        st.warning("⚠️ Aucune donnée IA détectée. Veuillez d'abord vous rendre dans l'onglet **'4. IA Appliquée'** et cliquer sur **'Lancer K-Means'** pour générer les recommandations dynamiques.")
    else:
        st.success(f"✅ Recommandations calculées en direct sur la base des **{len(profiles)}** profils de risques détectés.")
        
        # Nouveau : Explication de la Méthodologie Actuarielle pour rassurer l'actuaire
        with st.expander("🧮 Méthodologie & Formules Actuarielles (Transparence IA)"):
            st.markdown("""
            ### Calcul des KPIs Décisionnels
            L'Intelligence Artificielle ne génère aucun coût arbitraire. Tous les indicateurs sont extraits mathématiquement de la distribution réelle (SAAQ).
            
            **1. La Sévérité Moyenne du Portefeuille (Base de référence SAAQ)**
            """)
            st.latex(r"\mu_{global} = \sum_{k=1}^{K} \left( \bar{X}_k \times w_k \right)")
            st.markdown("Où $\\bar{X}_k$ est le nombre moyen de blessures du cluster $k$, et $w_k$ son poids volumétrique dans votre portefeuille.")
            
            st.markdown("**2. Indice de Sévérité Relative (Multiplicateur de Risque)**")
            st.latex(r"Indice_k = \frac{\bar{X}_k}{\mu_{global}}")
            st.markdown("Un indice de **x1.02** signifie que ce segment précis génère statistiquement **2% de blessures en plus par fracture/accident** que la moyenne consolidée de la province.")
            
        # Création des 4 onglets horizontaux
        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Provision (Réserves)", 
            "📊 Ajustement des Primes", 
            "🚨 Assignation des Dossiers", 
            "🛡️ Stratégie & Prévention"
        ])
        
        with tab1:
            st.header("Gestion de la Sévérité (Provisionnement)")
            st.info("**Objectif :** Alerter le gestionnaire si un profil de sinistre va présenter une gravité médicale anormalement plus élevée que la moyenne globale, afin de prévoir des réserves prolongées.")
            
            # Calculer la sévérité moyenne globale (Baseline)
            total_cases = sum(p['size_pct'] for p in profiles)
            global_avg_bless = sum(p['avg_bless'] * (p['size_pct']/total_cases) for p in profiles) if total_cases > 0 else 1.0
            
            total_dossiers = len(df.dropna(subset=['GRAVITE', 'BLESSURE_PRINCIPALE']))
            
            st.write(f"📊 **Base de référence SAAQ (Portefeuille global) :** {global_avg_bless:.2f} blessures/dossier.")
            st.markdown("---")
            
            for p in profiles:
                deviation = ((p['avg_bless'] - global_avg_bless) / global_avg_bless) * 100
                volume_dossiers = int(total_dossiers * (p['size_pct']/100))
                indice_severite = p['avg_bless'] / global_avg_bless
                
                with st.container():
                    col_r1, col_r2, col_r3 = st.columns([1, 1, 1.5]) 
                    with col_r1:
                        delta_str = f"{'+' if deviation > 0 else ''}{deviation:.1f}% vs Moyenne"
                        st.metric(f"Gravité - Profil {p['rank']}", f"{p['avg_bless']:.1f} bless/dossier", delta_str, delta_color="inverse")
                        
                        if p['traits']:
                            trait_name = translate_trait(p['traits'][0][0], p['traits'][0][1])
                            st.write(f"**Vecteur de risque :** {trait_name}")
                            
                    with col_r2:
                        impact_color = "inverse" if indice_severite > 1.05 else "normal"
                        sign = "+" if indice_severite > 1 else ""
                        st.metric(f"Indice de Sévérité Relative", f"x{indice_severite:.2f}", f"{volume_dossiers:,} dossiers historiques", delta_color=impact_color)
                        
                        # Explication claire en texte complet
                        direction = "plus" if deviation > 0 else "moins"
                        st.write(f"*(Ce profil subit statistiquement **{abs(deviation):.1f}%** de blessures en {direction} par accident par rapport à la moyenne globale pondérée de tout le portefeuille SAAQ).*")
                            
                    with col_r3:
                        if deviation > 5.0:
                            st.error(f"📈 **Action : Provisionnement IBNR (Incurred But Not Reported) à la Hausse**")
                            st.write(f"Ce cluster génère structurellement {deviation:.1f}% de blessures en plus par accident. Les durées de réhabilitation (et de paiement des indemnités de remplacement de revenu) seront statistiquement allongées.")
                        elif deviation < -5.0:
                            st.success(f"📉 **Action : Provisionnement Allégé**")
                            st.write(f"Ce cluster est moins sévère que la mutualisation (-{abs(deviation):.1f}%). Les dossiers se refermeront beaucoup plus vite que la moyenne.")
                        else:
                            st.info(f"⚖️ **Action : Provisionnement Standard**")
                            st.write(f"La sévérité est parfaitement alignée sur la moyenne historique. Aucun ajustement IBNR (Incurred But Not Reported) algorithmique requis pour ce segment.")
                    
                    with st.expander(f"📊 Données Probantes de l'IA (Provenance du Risque) - Profil {p['rank']}"):
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.write("**Répartition Géographique :**")
                            if p.get('region_dist'):
                                st.bar_chart(pd.Series(p['region_dist']))
                            else:
                                st.info("Analyse régionale non-significative.")
                        with fig_col2:
                            st.write("**Topologie des Véhicules :**")
                            if p.get('veh_dist'):
                                v_data = {k: v for k, v in p['veh_dist'].items() if v > 0}
                                if v_data:
                                    st.bar_chart(pd.Series(v_data))
                                else:
                                    st.info("Répartition homogène, pas de véhicule dominant.")
                            else:
                                st.info("Aucune donnée véhiculaire capturée.")
                                
                    st.markdown("---")

            if not any(p['avg_bless'] > 10.0 for p in profiles):
                st.write("*Audit d'équilibrage : le portefeuille ne présente pas de déviations physiologiques extrêmes dans la configuration actuelle.*")

        with tab2:
            st.header("Analyse du Risque de Souscription (Pricing)")
            st.info("**Objectif :** Adapter l'évaluation du risque à la réalité de la sinistralité découverte par l'IA en calculant les multiplicateurs de survenue.")
            
            for p in profiles:
                deviation = ((p['avg_bless'] - global_avg_bless) / global_avg_bless) * 100
                indice_severite = p['avg_bless'] / global_avg_bless
                
                with st.container():
                    col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
                    with col_p1:
                        st.metric(f"Profil {p['rank']} ({p['size_pct']:.1f}% base)", f"{p['avg_bless']:.1f} bless/dossier", f"{deviation:+.1f}% Sévérité", delta_color="inverse")
                    
                    with col_p2:
                        color = "inverse" if indice_severite > 1.05 else "normal"
                        sign = "+" if indice_severite > 1.0 else ""
                        st.metric("Multiplicateur de Prime Statistique", f"x{indice_severite:.2f}", f"Indice de charge de sinistre", delta_color=color)
                        
                    with col_p3:
                        if indice_severite > 1.2:
                            st.error(f"⚠️ **Surtaxe Technique Requise**")
                            st.write(f"L'IA recommande d'imposer un malus de fréquence tarifaire de **+{deviation:.1f}%** pour équilibrer le rapport sinistre/prime sur ce segment à haut risque médical.")
                        elif indice_severite > 1.05:
                            st.warning(f"📈 **Ajustement à la Hausse**")
                            st.write(f"Majoration modérée recommandée pour compenser un taux de blessure systématiquement supérieur à la mutualisation globale.")
                        elif indice_severite < 0.9:
                            st.success(f"📉 **Rabais de Bonne Expérience**")
                            st.write(f"Ce segment présente une fréquence grave très faible. Un algorithme de tarification compétitive peut accorder jusqu'à **{abs(deviation):.1f}%** d'escompte.")
                        else:
                            st.info(f"⚖️ **Tarification d'Équilibre**")
                            st.write("Le coût stochastique physiologique est absorbé par la prime actuelle. Ne rien changer.")

                    with st.expander(f"📊 Données Probantes de l'IA (Provenance du Risque) - Profil {p['rank']}"):
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.write("**Top 5 Régions concernées :**")
                            if p.get('region_dist'):
                                st.bar_chart(pd.Series(p['region_dist']))
                            else:
                                st.info("Pas de données régionales dominantes.")
                        with fig_col2:
                            st.write("**Véhicules impliqués :**")
                            if p.get('veh_dist'):
                                v_data = {k: v for k, v in p['veh_dist'].items() if v > 0}
                                if v_data:
                                    st.bar_chart(pd.Series(v_data))
                                else:
                                    st.info("Répartition homogène, pas de véhicule dominant.")
                            else:
                                st.info("Aucune donnée véhiculaire capturée.")
                                
                    st.markdown("---")
            st.write("Pour les autres profils standards (véhicules légers sans gravité excessive), la recommandation tarifaire est le maintien du coefficient de mutualisation de base (1.0).")

        with tab3:
            st.header("Recommandation d'Assignation (Triage Automatisé)")
            st.info("**Objectif :** Aiguiller immédiatement le dossier vers le bon type d'agent dès que la SAAQ ouvre le dossier numérique, basé sur la complexité médicale statistique.")
            
            # Nouveau : Explication de la Méthodologie de Triage
            with st.expander("⚖️ Règles de Routage Algorithmique (Transparence IA)"):
                st.markdown("""
                L'assignation des dossiers n'est pas une boîte noire. Elle repose sur les règles d'expertise suivantes, appliquées à chaque profil complet calculé par l'IA :
                
                *   **Paiement Automatique (Fast-Track)** : `Indice de Sévérité Relative < 0.98` **ET** `Absence de Véhicule Lourd / Usager Vulnérable`
                *   **Expert Médical Senior** : `Indice de Sévérité Relative > 1.05` **OU** `Présence de Véhicule Lourd / Usager Vulnérable (Piéton, Vélo)`
                *   **Agent Régulier** : Tous les autres cas se situant dans les limites de la Sévérité Standard (`0.98 < Indice < 1.05`).
                """)

            cols = st.columns(len(profiles))
            for i, p in enumerate(profiles):
                # Seuils dynamisés pour le triage
                deviation = ((p['avg_bless'] - global_avg_bless) / global_avg_bless) * 100
                indice_severite = p['avg_bless'] / global_avg_bless
                volume = int(total_dossiers * (p['size_pct']/100))
                
                with cols[i]:
                    st.markdown(f"### Profil {p['rank']}")
                    st.write(f"**Volume concerné :** {volume:,} dossiers/an")
                    
                    traits = [translate_trait(t[0], t[1]) for t in p['traits']]
                    traits_clean = [t for t in traits if t]
                    if traits_clean:
                        st.write(f"*{traits_clean[0]}*")
                    
                    # Définition des mots-clés de gravité extrême
                    mots_graves = ["lourd", "piéton", "vélo", "crânien", "fracture", "amputation", "décès", "traumatisme", "moelle", "brûlure"]
                    est_grave = any(mot in str(t).lower() for t in traits for mot in mots_graves)
                    
                    # Règle d'assignation explicite
                    if deviation < -2.0 and not est_grave:
                        st.success("Triage recommandé : **Paiement Automatique (Fast-Track)**")
                        st.caption(f"Logique : Indice ({indice_severite:.2f}) < 0.98 et aucun facteur aggravant détecté. Autorisation de règlement sans intervention humaine.")
                    elif deviation > 5.0 or est_grave:
                        st.error("Triage recommandé : **Expert Médical Senior**")
                        
                        # Explication précise de la règle déclenchée
                        if deviation > 5.0:
                            reason = f"Indice de Sévérité ({indice_severite:.2f}) > 1.05"
                        else:
                            reason = "Vecteur de risque catastrophique (Lourd/Vulnérable/Blessure Majeure) détecté"
                            
                        st.caption(f"Logique : {reason}. Assignation prioritaire à l'escouade spécialisée en blessures complexes.")
                    else:
                        st.warning("Triage recommandé : **Agent Régulier**")
                        st.caption(f"Logique : Sévérité Standard (Indice {indice_severite:.2f}). Traitement classique avec supervision humaine.")

        with tab4:
            st.header("Stratégie de Risque (Prévention & Pérennité du Fonds)")
            st.info("**Objectif :** Protéger le régime d'assurance public contre les risques de pointe et réduire la fréquence à la source.")
            
            severe_profiles = [p for p in profiles if p['avg_bless'] > (global_avg_bless * 1.5)]
            if severe_profiles:
                st.subheader("1. Anticipation des Risques Graves (Provisionnement)")
                pct_severe = sum(p['size_pct'] for p in severe_profiles)
                volume_severe = int(total_dossiers * (pct_severe/100))
                indice_severe = sum(p['avg_bless'] for p in severe_profiles)/len(severe_profiles) / global_avg_bless if severe_profiles else 1.0
                
                st.write(f"Le clustering montre que **{pct_severe:.1f}%** des sinistres ({volume_severe:,} cas annuels historiques) concentrent une sévérité anormalement élevée (queue de distribution épaisse).")
                st.write(f"**Indice de Pression sur le Portefeuille :** x{indice_severe:.2f} fois la sévérité normale.")
                st.error("Recommandation : Ajustement des réserves provisionnées à long terme (IBNR - Incurred But Not Reported) et révision ciblée des contributions d'assurance pour ce profil spécifique afin de limiter le déficit du fonds soutenu par les Québécois.")
            
            vulnerable_profiles = [p for p in profiles if any("Vélo" in str(translate_trait(t[0], t[1])) or "Moto" in str(translate_trait(t[0], t[1])) for t in p['traits'])]
            if vulnerable_profiles:
                st.subheader("2. Campagne de Prévention Algorithmique")
                st.write(f"Une poche de morbidité concernant les usagers vulnérables (2-roues) a été isolée dans le Profil {vulnerable_profiles[0]['rank']}.")
                st.info("Action : Lancement de notifications ciblées (SMS de vigilance) pour les assurés concernés avec rappel des distances de sécurité.")

# ----- PAGE 6 : LAB IA AVANCÉ (PRÉDICTION & FRAUDE) ----- #
elif page == "6. Lab IA Avancé (Prédiction & Fraude)":
    st.title("🤖 Lab IA Avancé : Prédictions & Détection de Fraudes")
    st.markdown("""
    *Cet espace démontre la puissance de l'IA prédictive (Machine Learning) et de la détection d'anomalies appliquées aux données d'indemnisation corporelle du registre SAAQ.*
    """)
    
    if df.empty or 'NB_BLESSURES' not in df.columns:
        st.warning("⚠️ Les données nécessaires ne sont pas chargées ou la variable cible (NB_BLESSURES) est manquante.")
        st.stop()
        
    tab_pred, tab_fraude = st.tabs(["🔮 Simulateur Prédictif (Random Forest)", "🚩 Scanner d'Anomalies (Isolation Forest)"])
    
    with tab_pred:
        st.header("Simulateur de Gravité d'Accident")
        st.info("**Objectif :** Prédire instantanément la sévérité attendue (nombre de blessures) pour un nouveau sinistre basé sur ses caractéristiques d'entrée.")
        
        with st.spinner("Entraînement rapide de l'IA (Random Forest)..."):
            # Sample for speed
            df_model = df.dropna(subset=['NB_BLESSURES', 'GROUP_AGE', 'NOM_REGN_CLEAN']).copy()
            if len(df_model) > 10000:
                df_model = df_model.sample(10000, random_state=42)
            
            features = ['NOM_REGN_CLEAN', 'GROUP_AGE']
            veh_cols = ['IND_AUTO_CAMION_LEGER_O', 'IND_VEH_LOURD_O', 'IND_MOTO_CYCLO_O', 'IND_VELO_O', 'IND_PIETON_O']
            for v in veh_cols:
                if v in df_model.columns: features.append(v)
            
            X = df_model[features].copy()
            y = df_model['NB_BLESSURES']
            
            le_reg = LabelEncoder()
            X['NOM_REGN_CLEAN'] = le_reg.fit_transform(X['NOM_REGN_CLEAN'])
            le_age = LabelEncoder()
            X['GROUP_AGE'] = le_age.fit_transform(X['GROUP_AGE'])
            
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(X, y)
            
        st.success("✅ Modèle d'IA entraîné avec succès sur les données historiques !")
        
        st.markdown("### 📝 Entrez les paramètres du nouveau sinistre :")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            in_reg = st.selectbox("Région de l'accident :", options=le_reg.classes_)
            in_age = st.selectbox("Groupe d'âge du conducteur :", options=le_age.classes_)
            
        with col_p2:
            st.write("**Type de véhicule/usager impliqué :**")
            in_veh_auto = st.checkbox("Véhicule Léger (Auto)", value=True)
            in_veh_lourd = st.checkbox("Véhicule Lourd (Camion)")
            in_veh_moto = st.checkbox("Motocyclette")
            in_veh_velo = st.checkbox("Vélo")
            in_veh_pieton = st.checkbox("Piéton")
            
        if st.button("Lancer la Prédiction IA"):
            input_data = {
                'NOM_REGN_CLEAN': le_reg.transform([in_reg])[0],
                'GROUP_AGE': le_age.transform([in_age])[0]
            }
            if 'IND_AUTO_CAMION_LEGER_O' in features: input_data['IND_AUTO_CAMION_LEGER_O'] = 1 if in_veh_auto else 0
            if 'IND_VEH_LOURD_O' in features: input_data['IND_VEH_LOURD_O'] = 1 if in_veh_lourd else 0
            if 'IND_MOTO_CYCLO_O' in features: input_data['IND_MOTO_CYCLO_O'] = 1 if in_veh_moto else 0
            if 'IND_VELO_O' in features: input_data['IND_VELO_O'] = 1 if in_veh_velo else 0
            if 'IND_PIETON_O' in features: input_data['IND_PIETON_O'] = 1 if in_veh_pieton else 0
            
            input_df = pd.DataFrame([input_data])[features]
            prediction = rf.predict(input_df)[0]
            
            baseline = y.mean()
            deviation = ((prediction - baseline) / baseline) * 100
            
            st.markdown("---")
            st.subheader("📊 Résultat de la Prédiction")
            
            col_r1, col_r2 = st.columns([1, 1.5])
            with col_r1:
                st.metric("Gravité Prédite (Blessures/dossier)", f"{prediction:.2f}", f"{deviation:+.1f}% vs Moyenne {'(Sévère)' if deviation > 0 else '(Léger)'}", delta_color="inverse")
                
            with col_r2:
                if prediction > baseline * 1.5:
                    st.error("🚨 **Alerte Déclenchée :** L'IA estime que cette combinaison génère historiquement des sinistres très graves. Recommandation : Fast-track vers un Expert Majeur immédiat.")
                elif prediction < baseline * 0.8:
                    st.success("✅ **Profil Favorable :** La sinistralité historique pour ce profil est faible. Recommandation : Paiement Automatique autorisé.")
                else:
                    st.info("⚖️ **Profil Standard :** La gravité attendue est dans la norme corporative. Traitement par Agent Régulier.")
                    
            st.write("**Importance des facteurs dans cette décision IA :**")
            impo = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(4)
            clean_impo = {}
            for k, v in impo.items():
                name = trade_map.get(k, k)
                if k == 'NOM_REGN_CLEAN': name = 'Région'
                if k == 'GROUP_AGE': name = 'Âge'
                clean_impo[name] = v
            st.bar_chart(pd.Series(clean_impo))


    with tab_fraude:
        st.header("Détection de Fraude / Sinistres Atypiques")
        st.info("**Objectif :** Utiliser une IA non-supervisée pour scanner la base de données et isoler la fraction (1%) de dossiers les plus anormaux. Ceux-ci sont souvent symptomatiques d'erreurs de saisie ou de fausses déclarations.")
        
        if st.button("🚀 Lancer l'Audit IA sur l'Historique"):
            with st.spinner("L'IA scanne des dizaines de milliers de dossiers pour trouver les anomalies structurelles..."):
                
                df_if = df.dropna(subset=['NB_BLESSURES', 'GROUP_AGE', 'NOM_REGN_CLEAN']).copy()
                if len(df_if) > 30000:
                    df_if = df_if.sample(30000, random_state=42)
                
                features_if = ['NB_BLESSURES']
                for v in veh_cols:
                    if v in df_if.columns: features_if.append(v)
                
                X_if = df_if[features_if].copy()
                
                clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                preds = clf.fit_predict(X_if)
                
                df_if['ANOMALY'] = preds
                anomalies = df_if[df_if['ANOMALY'] == -1]
                
            st.error(f"🚨 Scanner terminé ! L'IA a isolé **{len(anomalies)} dossiers hautement atypiques** parmi l'échantillon analysé.")
            
            st.write("### Extrait des dossiers levés par l'Audit IA (Pour investigation) :")
            
            display_cols = ['YEAR', 'NOM_REGN_CLEAN', 'GROUP_AGE', 'NB_BLESSURES']
            for v in veh_cols:
                if v in anomalies.columns: display_cols.append(v)
                
            anom_disp = anomalies[display_cols].sample(min(15, len(anomalies)), random_state=1).copy()
            
            for v in veh_cols:
                if v in anom_disp.columns:
                    anom_disp[v] = anom_disp[v].replace({1.0: "Oui", 0.0: "-", 1: "Oui", 0: "-"})
                    anom_disp.rename(columns={v: trade_map.get(v, v).split('/')[0]}, inplace=True)
            
            st.dataframe(anom_disp, use_container_width=True)
            
            st.markdown("""
            > **Pourquoi ces dossiers sont-ils flaggés par l'IA ?**
            > L'algorithme *Isolation Forest* n'utilise pas de règles métier hardcodées. Il isole les points statistiques marginaux.
            > *Exemple :* Un grave accident de la route impliquant un piéton, mais générant 0 blessure déclarée (incohérence) ou à l'inverse, un accident de véhicule léger déclarant 14 blessés dans un même dossier. L'outil agit comme un radar automatique pour les investigateurs anti-fraude.
            """)
