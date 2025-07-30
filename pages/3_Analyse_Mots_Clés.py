import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from textblob import TextBlob
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Analyse des Mots-Clés", page_icon="🔤", layout="wide")

# CSS personnalisé
st.markdown(
    """
<style>
    .keyword-positive { 
        background: #e8f5e8; 
        color: #2e7d32; 
        padding: 0.3rem 0.6rem; 
        border-radius: 15px; 
        margin: 0.2rem;
        display: inline-block;
    }
    .keyword-negative { 
        background: #ffebee; 
        color: #d32f2f; 
        padding: 0.3rem 0.6rem; 
        border-radius: 15px; 
        margin: 0.2rem;
        display: inline-block;
    }
    .keyword-neutral { 
        background: #fff3e0; 
        color: #f57c00; 
        padding: 0.3rem 0.6rem; 
        border-radius: 15px; 
        margin: 0.2rem;
        display: inline-block;
    }
    .theme-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f4e79;
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
        mots_negatifs.columns = mots_negatifs.columns.str.strip()
        mots_positifs.columns = mots_positifs.columns.str.strip()
        mots_neutres.columns = mots_neutres.columns.str.strip()

        return avis_df, mots_negatifs, mots_positifs, mots_neutres
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None, None, None, None


def create_word_frequency_chart(words_df, title, color):
    """Crée un graphique de fréquence des mots"""
    if words_df is None or words_df.empty:
        return None

    # Prendre les colonnes disponibles
    if len(words_df.columns) >= 2:
        mot_col = words_df.columns[0]
        freq_col = words_df.columns[1]
    else:
        return None

    # Prendre les top 20
    top_words = words_df.head(20)

    fig = px.bar(
        top_words,
        x=freq_col,
        y=mot_col,
        orientation="h",
        title=title,
        labels={freq_col: "Fréquence", mot_col: "Mots"},
        color_discrete_sequence=[color],
    )

    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})

    return fig


def create_comparative_word_chart(mots_positifs, mots_negatifs, mots_neutres):
    """Crée un graphique comparatif des mots par sentiment"""

    data_combined = []

    # Préparer les données pour chaque sentiment
    for df, sentiment, color in [
        (mots_positifs, "Positif", "#2e7d32"),
        (mots_negatifs, "Négatif", "#d32f2f"),
        (mots_neutres, "Neutre", "#f57c00"),
    ]:
        if df is not None and not df.empty and len(df.columns) >= 2:
            top_10 = df.head(10)
            for _, row in top_10.iterrows():
                data_combined.append(
                    {
                        "mot": row.iloc[0],
                        "frequence": row.iloc[1],
                        "sentiment": sentiment,
                        "color": color,
                    }
                )

    if not data_combined:
        return None

    df_combined = pd.DataFrame(data_combined)

    fig = px.bar(
        df_combined,
        x="frequence",
        y="mot",
        color="sentiment",
        orientation="h",
        title="Top 10 Mots par Sentiment",
        color_discrete_map={
            "Positif": "#2e7d32",
            "Négatif": "#d32f2f",
            "Neutre": "#f57c00",
        },
    )

    fig.update_layout(height=800)

    return fig


def create_word_cloud_advanced(words_df, title, colormap="viridis"):
    """Crée un nuage de mots avancé"""
    if words_df is None or words_df.empty:
        return None

    # Créer le dictionnaire de fréquences
    if len(words_df.columns) >= 2:
        word_freq = dict(zip(words_df.iloc[:, 0], words_df.iloc[:, 1]))
    else:
        return None

    # Générer le nuage de mots
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        max_font_size=80,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=20, pad=20)

    return fig


def analyze_keyword_evolution(avis_df, keywords):
    """Analyse l'évolution des mots-clés dans le temps"""
    if "date" not in avis_df.columns or "commentaire" not in avis_df.columns:
        return None

    # Créer un DataFrame pour l'évolution
    evolution_data = []

    # Grouper par mois
    avis_df["mois"] = avis_df["date"].dt.to_period("M")

    for keyword in keywords:
        monthly_counts = []
        for period in avis_df["mois"].unique():
            if pd.notna(period):
                period_comments = avis_df[avis_df["mois"] == period][
                    "commentaire"
                ].fillna("")
                count = sum(
                    keyword.lower() in comment.lower() for comment in period_comments
                )
                monthly_counts.append(
                    {"mois": str(period), "mot": keyword, "occurrences": count}
                )

        evolution_data.extend(monthly_counts)

    if not evolution_data:
        return None

    evolution_df = pd.DataFrame(evolution_data)

    fig = px.line(
        evolution_df,
        x="mois",
        y="occurrences",
        color="mot",
        title="Évolution des Mots-Clés dans le Temps",
        labels={"occurrences": "Nombre d'occurrences", "mois": "Mois"},
    )

    fig.update_layout(height=500)

    return fig


def create_keyword_network(words_df, sentiment_color):
    """Crée un réseau de mots-clés"""
    if words_df is None or words_df.empty or len(words_df) < 3:
        return None

    # Créer un graphique de réseau simple
    G = nx.Graph()

    # Ajouter les nœuds (mots)
    top_words = words_df.head(15)
    for _, row in top_words.iterrows():
        word = row.iloc[0]
        freq = row.iloc[1]
        G.add_node(word, size=freq)

    # Ajouter des arêtes basées sur la co-occurrence (simulation)
    words_list = list(top_words.iloc[:, 0])
    for i in range(len(words_list)):
        for j in range(
            i + 1, min(i + 4, len(words_list))
        ):  # Connecter aux 3 mots suivants
            G.add_edge(words_list[i], words_list[j])

    # Calculer les positions
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Préparer les données pour Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Créer le graphique
    fig = go.Figure()

    # Ajouter les arêtes
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
    )

    # Ajouter les nœuds
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]["size"])

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            hoverinfo="text",
            marker=dict(
                size=[s / 10 for s in node_size],  # Normaliser la taille
                color=sentiment_color,
                line=dict(width=2, color="white"),
            ),
        )
    )

    fig.update_layout(
        title="Réseau de Mots-Clés",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Taille des nœuds proportionnelle à la fréquence",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,  # finalized from here
    )
    return fig


# Chargement des données
avis_df, mots_negatifs, mots_positifs, mots_neutres = load_data()

# Si les données sont chargées avec succès
if avis_df is not None:
    # --- ANALYSE DES MOTS-CLÉS ---
    st.title("Analyse des Mots-Clés")

    # Sélection des mots-clés par sentiment
    sentiment_selection = st.radio(
        "Afficher les mots-clés pour quel type de sentiment ?",
        ("Positifs", "Négatifs", "Neutres"),
    )

    if sentiment_selection == "Positifs":
        mots_clefs = mots_positifs
        couleur = "#2e7d32"
    elif sentiment_selection == "Négatifs":
        mots_clefs = mots_negatifs
        couleur = "#d32f2f"
    else:
        mots_clefs = mots_neutres
        couleur = "#f57c00"

    # Graphique de fréquence des mots
    st.subheader(f"Fréquence des Mots-Clés ({sentiment_selection})")
    fig = create_word_frequency_chart(
        mots_clefs, f"Top Mots-Clés {sentiment_selection}", couleur
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # Nuage de mots
    st.subheader(f"Nuage de Mots-Clés {sentiment_selection}")
    fig = create_word_cloud_advanced(
        mots_clefs,
        f"Nuage de Mots-Clés {sentiment_selection}",
        colormap="RdYlGn" if sentiment_selection == "Positifs" else "RdYlBu",
    )
    if fig is not None:
        st.pyplot(fig)

    # Réseau de mots-clés
    st.subheader(f"Réseau de Mots-Clés {sentiment_selection}")
    fig = create_keyword_network(mots_clefs, couleur)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # --- ANALYSE COMPARATIVE ---
    st.title("Analyse Comparative des Mots-Clés")

    # Graphique comparatif des mots par sentiment
    st.subheader("Comparaison des Mots-Clés par Sentiment")
    fig = create_comparative_word_chart(mots_positifs, mots_negatifs, mots_neutres)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    # --- EVOLUTION DES MOTS-CLÉS ---
    st.title("Évolution des Mots-Clés")

    # Sélecteur de mots-clés
    keywords = st.multiselect(
        "Sélectionnez les mots-clés à analyser",
        options=mots_positifs.iloc[:, 0].tolist()
        + mots_negatifs.iloc[:, 0].tolist()
        + mots_neutres.iloc[:, 0].tolist(),
        default=mots_positifs.iloc[:5, 0].tolist()
        + mots_negatifs.iloc[:5, 0].tolist()
        + mots_neutres.iloc[:5, 0].tolist(),
    )

    if keywords:
        # Graphique d'évolution des mots-clés
        st.subheader("Évolution Mensuelle des Mots-Clés")
        fig = analyze_keyword_evolution(avis_df, keywords)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
