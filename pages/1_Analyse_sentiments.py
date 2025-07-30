import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Configuration de la page
st.set_page_config(page_title="Analyse des Sentiments", page_icon="📊", layout="wide")

# CSS personnalisé
st.markdown(
    """
<style>
    .sentiment-positive { color: #2e7d32; font-weight: bold; }
    .sentiment-negative { color: #d32f2f; font-weight: bold; }
    .sentiment-neutral { color: #f57c00; font-weight: bold; }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Charge les données"""
    try:
        avis_df = pd.read_csv("avis_titan_complet.csv")
        mots_negatifs = pd.read_csv("top_mots_negatifs_titan.csv")
        mots_positifs = pd.read_csv("top_mots_positifs_titan.csv")
        mots_neutres = pd.read_csv("top_mots_neutres_titan.csv")

        # Nettoyage des colonnes
        avis_df.columns = avis_df.columns.str.strip()

        if "date" in avis_df.columns:
            avis_df["date"] = pd.to_datetime(avis_df["date"], errors="coerce")

        return avis_df, mots_negatifs, mots_positifs, mots_neutres
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None, None, None, None


def create_sentiment_pie_chart(avis_df):
    """Crée un graphique en secteurs pour la distribution des sentiments"""
    if "sentiment" not in avis_df.columns:
        return None

    sentiment_counts = avis_df["sentiment"].value_counts()

    colors = {"positif": "#2e7d32", "negatif": "#d32f2f", "neutre": "#f57c00"}

    fig = go.Figure(
        data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker=dict(
                    colors=[
                        colors.get(label, "#1f77b4") for label in sentiment_counts.index
                    ]
                ),
            )
        ]
    )

    fig.update_layout(title="Distribution des Sentiments", height=400, showlegend=True)

    return fig


def create_sentiment_evolution(avis_df):
    """Crée un graphique d'évolution des sentiments dans le temps"""
    if "date" not in avis_df.columns or "sentiment" not in avis_df.columns:
        return None

    # Grouper par mois et sentiment
    avis_df["mois"] = avis_df["date"].dt.to_period("M")
    sentiment_evolution = (
        avis_df.groupby(["mois", "sentiment"]).size().unstack(fill_value=0)
    )

    fig = go.Figure()

    colors = {"positif": "#2e7d32", "negatif": "#d32f2f", "neutre": "#f57c00"}

    for sentiment in sentiment_evolution.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_evolution.index.astype(str),
                y=sentiment_evolution[sentiment],
                mode="lines+markers",
                name=sentiment.capitalize(),
                line=dict(color=colors.get(sentiment, "#1f77b4"), width=3),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title="Évolution des Sentiments dans le Temps",
        xaxis_title="Mois",
        yaxis_title="Nombre d'avis",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_sentiment_heatmap(avis_df):
    """Crée une heatmap des sentiments par jour de la semaine et heure"""
    if "date" not in avis_df.columns or "sentiment" not in avis_df.columns:
        return None

    # Extraction jour de la semaine et heure
    avis_df["jour_semaine"] = avis_df["date"].dt.day_name()
    avis_df["heure"] = avis_df["date"].dt.hour

    # Pivot pour la heatmap
    heatmap_data = avis_df.pivot_table(
        index="jour_semaine",
        columns="heure",
        values="sentiment",
        aggfunc="count",
        fill_value=0,
    )

    fig = px.imshow(
        heatmap_data,
        title="Heatmap des Avis par Jour et Heure",
        labels=dict(x="Heure", y="Jour de la semaine", color="Nombre d'avis"),
        color_continuous_scale="YlOrRd",
    )

    return fig


def create_word_cloud(words_df, title, color_scheme="viridis"):
    """Crée un nuage de mots"""
    if words_df is None or words_df.empty:
        return None

    # Supposons que le DataFrame a une colonne 'mot' et 'frequence'
    if "mot" in words_df.columns and "frequence" in words_df.columns:
        word_freq = dict(zip(words_df["mot"], words_df["frequence"]))
    else:
        # Prendre les deux premières colonnes
        word_freq = dict(zip(words_df.iloc[:, 0], words_df.iloc[:, 1]))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=color_scheme,
        max_words=100,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    return fig


def create_top_words_bar_chart(words_df, title, color):
    """Crée un graphique en barres pour les mots les plus fréquents"""
    if words_df is None or words_df.empty:
        return None

    # Prendre les 15 premiers mots
    top_words = words_df.head(15)

    if "mot" in words_df.columns and "frequence" in words_df.columns:
        x_data = top_words["mot"]
        y_data = top_words["frequence"]
    else:
        x_data = top_words.iloc[:, 0]
        y_data = top_words.iloc[:, 1]

    fig = px.bar(
        x=y_data,
        y=x_data,
        orientation="h",
        title=title,
        labels={"x": "Fréquence", "y": "Mots"},
        color_discrete_sequence=[color],
    )

    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})

    return fig


def sentiment_distribution_by_category(avis_df):
    """Analyse la distribution des sentiments par catégorie"""
    if "categorie" not in avis_df.columns or "sentiment" not in avis_df.columns:
        return None

    # Créer un graphique en barres empilées
    sentiment_cat = pd.crosstab(avis_df["categorie"], avis_df["sentiment"])

    fig = px.bar(
        sentiment_cat,
        title="Distribution des Sentiments par Catégorie",
        labels={"value": "Nombre d'avis", "index": "Catégorie"},
        color_discrete_map={
            "positif": "#2e7d32",
            "negatif": "#d32f2f",
            "neutre": "#f57c00",
        },
    )

    fig.update_layout(height=500)
    return fig


def create_sentiment_comparison_chart(mots_positifs, mots_negatifs, mots_neutres):
    """Crée un graphique de comparaison des fréquences par sentiment"""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Mots Positifs", "Mots Négatifs", "Mots Neutres"),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
        ],
    )

    datasets = [
        (mots_positifs, "#2e7d32", 1),
        (mots_negatifs, "#d32f2f", 2),
        (mots_neutres, "#f57c00", 3),
    ]

    for words_df, color, col in datasets:
        if words_df is not None and not words_df.empty:
            top_10 = words_df.head(10)
            if "mot" in words_df.columns and "frequence" in words_df.columns:
                x_data = top_10["mot"]
                y_data = top_10["frequence"]
            else:
                x_data = top_10.iloc[:, 0]
                y_data = top_10.iloc[:, 1]

            fig.add_trace(
                go.Bar(x=x_data, y=y_data, marker_color=color, showlegend=False),
                row=1,
                col=col,
            )

    fig.update_layout(
        title="Comparaison des Mots les Plus Fréquents par Sentiment", height=500
    )

    return fig


def main():
    st.title("📊 Analyse Détaillée des Sentiments")

    # Chargement des données
    with st.spinner("Chargement des données..."):
        avis_df, mots_negatifs, mots_positifs, mots_neutres = load_data()

    if avis_df is None:
        st.error("Impossible de charger les données")
        return

    # Sidebar avec filtres
    st.sidebar.header("🔍 Filtres")

    # Filtre par période
    if "date" in avis_df.columns:
        date_range = st.sidebar.date_input(
            "Période",
            value=(avis_df["date"].min().date(), avis_df["date"].max().date()),
        )

        # Filtrer les données
        mask = (avis_df["date"].dt.date >= date_range[0]) & (
            avis_df["date"].dt.date <= date_range[1]
        )
        avis_filtered = avis_df[mask]
    else:
        avis_filtered = avis_df

    # Filtre par sentiment
    if "sentiment" in avis_df.columns:
        selected_sentiments = st.sidebar.multiselect(
            "Sentiments",
            options=avis_df["sentiment"].unique(),
            default=avis_df["sentiment"].unique(),
        )
        avis_filtered = avis_filtered[
            avis_filtered["sentiment"].isin(selected_sentiments)
        ]

    # Métriques principales
    st.header("📈 Métriques de Sentiment")

    col1, col2, col3, col4 = st.columns(4)

    if "sentiment" in avis_filtered.columns:
        sentiment_counts = avis_filtered["sentiment"].value_counts()
        total_avis = len(avis_filtered)

        with col1:
            positifs = sentiment_counts.get("positif", 0)
            st.metric(
                "Avis Positifs", f"{positifs:,}", f"{(positifs/total_avis)*100:.1f}%"
            )

        with col2:
            negatifs = sentiment_counts.get("negatif", 0)
            st.metric(
                "Avis Négatifs", f"{negatifs:,}", f"{(negatifs/total_avis)*100:.1f}%"
            )

        with col3:
            neutres = sentiment_counts.get("neutre", 0)
            st.metric(
                "Avis Neutres", f"{neutres:,}", f"{(neutres/total_avis)*100:.1f}%"
            )

        with col4:
            if positifs > 0 and negatifs > 0:
                ratio = positifs / negatifs
                st.metric(
                    "Ratio Pos/Neg",
                    f"{ratio:.2f}",
                    "Excellent" if ratio > 2 else "Bon" if ratio > 1 else "À améliorer",
                )

    # Graphique en secteurs pour la distribution
    pie_fig = create_sentiment_pie_chart(avis_filtered)
    if pie_fig:
        st.plotly_chart(pie_fig, use_container_width=True)

    # Graphiques d'analyse
    st.header("📊 Analyses Visuelles")

    # Évolution temporelle
    evolution_fig = create_sentiment_evolution(avis_filtered)
    if evolution_fig:
        st.plotly_chart(evolution_fig, use_container_width=True)

    # Heatmap
    heatmap_fig = create_sentiment_heatmap(avis_filtered)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Distribution par catégorie
    category_fig = sentiment_distribution_by_category(avis_filtered)
    if category_fig:
        st.plotly_chart(category_fig, use_container_width=True)

    # Nuages de mots
    st.header("☁️ Nuages de Mots")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Mots Positifs")
        if mots_positifs is not None:
            fig_pos = create_word_cloud(mots_positifs, "Mots Positifs", "Greens")
            if fig_pos:
                st.pyplot(fig_pos)
            else:
                st.info("Aucune donnée disponible pour les mots positifs")

    with col2:
        st.subheader("Mots Négatifs")
        if mots_negatifs is not None:
            fig_neg = create_word_cloud(mots_negatifs, "Mots Négatifs", "Reds")
            if fig_neg:
                st.pyplot(fig_neg)
            else:
                st.info("Aucune donnée disponible pour les mots négatifs")

    with col3:
        st.subheader("Mots Neutres")
        if mots_neutres is not None:
            fig_neu = create_word_cloud(mots_neutres, "Mots Neutres", "Oranges")
            if fig_neu:
                st.pyplot(fig_neu)
            else:
                st.info("Aucune donnée disponible pour les mots neutres")

    # Graphique de comparaison des mots
    comparison_fig = create_sentiment_comparison_chart(
        mots_positifs, mots_negatifs, mots_neutres
    )
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)

    # Analyse détaillée des mots-clés
    st.header("🔍 Analyse des Mots-Clés")

    tab1, tab2, tab3 = st.tabs(["Positifs", "Négatifs", "Neutres"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            if mots_positifs is not None:
                st.dataframe(mots_positifs, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        with col2:
            bar_fig_pos = create_top_words_bar_chart(
                mots_positifs, "Top 15 Mots Positifs", "#2e7d32"
            )
            if bar_fig_pos:
                st.plotly_chart(bar_fig_pos, use_container_width=True)

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            if mots_negatifs is not None:
                st.dataframe(mots_negatifs, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        with col2:
            bar_fig_neg = create_top_words_bar_chart(
                mots_negatifs, "Top 15 Mots Négatifs", "#d32f2f"
            )
            if bar_fig_neg:
                st.plotly_chart(bar_fig_neg, use_container_width=True)

    with tab3:
        col1, col2 = st.columns([1, 2])
        with col1:
            if mots_neutres is not None:
                st.dataframe(mots_neutres, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        with col2:
            bar_fig_neu = create_top_words_bar_chart(
                mots_neutres, "Top 15 Mots Neutres", "#f57c00"
            )
            if bar_fig_neu:
                st.plotly_chart(bar_fig_neu, use_container_width=True)

    # Statistiques avancées
    st.header("📊 Statistiques Avancées")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistiques des Mots")
        if all(df is not None for df in [mots_positifs, mots_negatifs, mots_neutres]):
            stats_data = {
                "Sentiment": ["Positif", "Négatif", "Neutre"],
                "Nombre de mots uniques": [
                    len(mots_positifs),
                    len(mots_negatifs),
                    len(mots_neutres),
                ],
                "Fréquence moyenne": [
                    mots_positifs.iloc[:, 1].mean() if not mots_positifs.empty else 0,
                    mots_negatifs.iloc[:, 1].mean() if not mots_negatifs.empty else 0,
                    mots_neutres.iloc[:, 1].mean() if not mots_neutres.empty else 0,
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    with col2:
        st.subheader("Répartition des Avis")
        if "sentiment" in avis_filtered.columns:
            sentiment_stats = avis_filtered["sentiment"].value_counts()
            fig_stats = px.bar(
                x=sentiment_stats.index,
                y=sentiment_stats.values,
                title="Nombre d'avis par sentiment",
                color=sentiment_stats.index,
                color_discrete_map={
                    "positif": "#2e7d32",
                    "negatif": "#d32f2f",
                    "neutre": "#f57c00",
                },
            )
            try:
                st.plotly_chart(fig_stats, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur d'affichage du graphique: {e}")


if __name__ == "__main__":
    main()
