import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Avis Clients - Banque Titan",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
    }
    
    .kpi-positive {
        color: #2e7d32;
        font-weight: bold;
    }
    
    .kpi-negative {
        color: #d32f2f;
        font-weight: bold;
    }
    
    .kpi-neutral {
        color: #f57c00;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge et prépare les données"""
    try:
        # Chargement des fichiers
        avis_df = pd.read_csv("avis_titan_complet.csv")
        mots_negatifs = pd.read_csv("top_mots_negatifs_titan.csv")
        mots_positifs = pd.read_csv("top_mots_positifs_titan.csv")
        mots_neutres = pd.read_csv("top_mots_neutres_titan.csv")

        # Nettoyage des colonnes (suppression des espaces)
        avis_df.columns = avis_df.columns.str.strip()

        # Conversion de la date si elle existe
        if "date" in avis_df.columns:
            avis_df["date"] = pd.to_datetime(avis_df["date"], errors="coerce")

        return avis_df, mots_negatifs, mots_positifs, mots_neutres

    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None, None, None, None


# Fonction pour calculer les KPIs
def calculate_kpis(avis_df):
    """Calcule les KPIs principaux"""
    if avis_df is None:
        return {}

    kpis = {}

    # KPIs de base
    kpis["total_avis"] = len(avis_df)

    # Analyse des sentiments (basée sur la colonne 'sentiment_class')
    if "sentiment_class" in avis_df.columns:
        sentiment_counts = avis_df["sentiment_class"].str.lower().value_counts()
        kpis["positifs"] = sentiment_counts.get("positif", 0)
        kpis["negatifs"] = sentiment_counts.get("négatif", 0) + sentiment_counts.get(
            "negatif", 0
        )
        kpis["neutres"] = sentiment_counts.get("neutre", 0)

    # Score de satisfaction moyen
    if "note_avis" in avis_df.columns:
        kpis["score_moyen"] = avis_df["note_avis"].mean()
        kpis["score_median"] = avis_df["note_avis"].median()

    # Évolution temporelle
    if "date" in avis_df.columns:
        avis_df["mois"] = avis_df["date"].dt.to_period("M")
        kpis["evolution_mensuelle"] = avis_df.groupby("mois").size()

    return kpis


# Fonction pour créer des graphiques
def create_sentiment_chart(kpis):
    """Crée un graphique de distribution des sentiments"""
    if not all(k in kpis for k in ["positifs", "negatifs", "neutres"]):
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Positifs", "Négatifs", "Neutres"],
                y=[kpis["positifs"], kpis["negatifs"], kpis["neutres"]],
                marker_color=["#2e7d32", "#d32f2f", "#f57c00"],
            )
        ]
    )

    fig.update_layout(
        title="Distribution des Sentiments",
        xaxis_title="Sentiment",
        yaxis_title="Nombre d'avis",
        showlegend=False,
        height=400,
    )

    return fig


def create_score_distribution(avis_df):
    """Crée un histogramme des scores"""
    if "note" not in avis_df.columns:
        return None

    fig = px.histogram(
        avis_df,
        x="note",
        nbins=5,
        title="Distribution des Notes",
        labels={"note": "Note", "count": "Nombre d'avis"},
    )

    fig.update_layout(height=400)
    return fig


# Interface principale
def main():
    # En-tête
    st.markdown(
        '<div class="main-header">🏦 Dashboard Avis Clients - Banque Titan</div>',
        unsafe_allow_html=True,
    )

    # Chargement des données
    with st.spinner("Chargement des données..."):
        avis_df, mots_negatifs, mots_positifs, mots_neutres = load_data()

    if avis_df is None:
        st.error(
            "Impossible de charger les données. Vérifiez que les fichiers CSV sont présents."
        )
        return

    # Calcul des KPIs
    kpis = calculate_kpis(avis_df)

    # Sidebar avec filtres
    st.sidebar.header("🔍 Filtres")

    # Filtre par période
    if "date" in avis_df.columns:
        date_range = st.sidebar.date_input(
            "Période d'analyse",
            value=(avis_df["date"].min().date(), avis_df["date"].max().date()),
            min_value=avis_df["date"].min().date(),
            max_value=avis_df["date"].max().date(),
        )

    # Filtre par sentiment
    if "sentiment" in avis_df.columns:
        sentiments = st.sidebar.multiselect(
            "Sentiments",
            options=avis_df["sentiment"].unique(),
            default=avis_df["sentiment"].unique(),
        )

    # KPIs principaux
    st.header("📊 KPIs Principaux")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Avis", value=f"{kpis.get('total_avis', 0):,}", delta=None
        )

    with col2:
        if "score_moyen" in kpis:
            st.metric(
                label="Score Moyen", value=f"{kpis['score_moyen']:.2f}/5", delta=None
            )

    with col3:
        if "positifs" in kpis:
            pourcentage_positif = (kpis["positifs"] / kpis["total_avis"]) * 100
            st.metric(
                label="Avis Positifs",
                value=f"{kpis['positifs']:,}",
                delta=f"{pourcentage_positif:.1f}%",
            )

    with col4:
        if "negatifs" in kpis:
            pourcentage_negatif = (kpis["negatifs"] / kpis["total_avis"]) * 100
            st.metric(
                label="Avis Négatifs",
                value=f"{kpis['negatifs']:,}",
                delta=f"-{pourcentage_negatif:.1f}%",
            )

    # Graphiques principaux
    st.header("📈 Analyses Visuelles")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique des sentiments
        sentiment_chart = create_sentiment_chart(kpis)
        if sentiment_chart:
            st.plotly_chart(sentiment_chart, use_container_width=True)

    with col2:
        # Distribution des scores
        score_chart = create_score_distribution(avis_df)
        if score_chart:
            st.plotly_chart(score_chart, use_container_width=True)

    # Évolution temporelle
    if "evolution_mensuelle" in kpis:
        st.header("📅 Évolution Temporelle")

        evolution_df = kpis["evolution_mensuelle"].reset_index()
        evolution_df["mois"] = evolution_df["mois"].astype(str)

        fig = px.line(
            evolution_df,
            x="mois",
            y=0,
            title="Évolution du Nombre d'Avis par Mois",
            labels={"0": "Nombre d'avis", "mois": "Mois"},
        )

        st.plotly_chart(fig, use_container_width=True)

    # Analyse des mots-clés
    st.header("🔤 Analyse des Mots-Clés")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔴 Mots Négatifs")
        if mots_negatifs is not None and not mots_negatifs.empty:
            st.dataframe(mots_negatifs.head(10), use_container_width=True)

    with col2:
        st.subheader("🟢 Mots Positifs")
        if mots_positifs is not None and not mots_positifs.empty:
            st.dataframe(mots_positifs.head(10), use_container_width=True)

    with col3:
        st.subheader("🟡 Mots Neutres")
        if mots_neutres is not None and not mots_neutres.empty:
            st.dataframe(mots_neutres.head(10), use_container_width=True)

    # Tableau détaillé
    st.header("📋 Données Détaillées")

    # Affichage des données brutes avec pagination
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(avis_df.head(100), use_container_width=True)

    # Statistiques descriptives
    st.header("📊 Statistiques Descriptives")

    if "note" in avis_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution des Notes")
            stats = avis_df["note"].describe()
            st.write(stats)

        with col2:
            st.subheader("Répartition par Sentiment")
            if "sentiment" in avis_df.columns:
                sentiment_stats = avis_df["sentiment"].value_counts()
                st.write(sentiment_stats)


if __name__ == "__main__":
    main()