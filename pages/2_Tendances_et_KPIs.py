import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Dashboard Avis Nickel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration de la connexion à la base de données Neon ---
@st.cache_resource
def init_connection():
    db_config = st.secrets["connections"]["postgresql"]
    return psycopg2.connect(**db_config)

conn = init_connection()

# --- Fonction pour charger les données (avec mise en cache) ---
@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=colnames)

# --- Fonctions utilitaires pour les calculs ---
def format_timedelta(td):
    if pd.isna(td):
        return "N/A"
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}"

# --- Barre latérale pour les filtres ---
st.sidebar.header("Filtres")

tz = pytz.timezone("Europe/Paris")
now_tz = tz.localize(datetime.now()) # Assurez-vous que now_tz est défini et aware

# --- Choix du mode de filtre ---
filter_mode = st.sidebar.radio(
    "Comment voulez-vous filtrer les dates ?",
    ("Période prédéfinie", "Plage de dates personnalisée")
)

# Initialisation des dates de filtre
start_date_filter = None
end_date_filter = None

if filter_mode == "Période prédéfinie":
    # Options pour le menu déroulant de la période
    time_options = {
        "Hier": (now_tz - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 7 derniers jours": (now_tz - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 15 derniers jours": (now_tz - timedelta(days=15)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 30 derniers jours": (now_tz - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 2 derniers mois": (now_tz - timedelta(days=60)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 3 derniers mois": (now_tz - timedelta(days=90)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 6 derniers mois": (now_tz - timedelta(days=180)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Ces 12 derniers mois": (now_tz - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0),
        "Depuis le début": datetime(2000, 1, 1, tzinfo=tz) # Une date très ancienne, ajustez si nécessaire
    }

    selected_period_label = st.sidebar.selectbox(
        "Sélectionnez une période prédéfinie :",
        list(time_options.keys()),
        index=7 # Sélectionne "Ces 12 derniers mois" par défaut
    )

    start_date_filter = time_options[selected_period_label]
    end_date_filter = now_tz.replace(hour=23, minute=59, second=59, microsecond=999999) # Fin de la journée actuelle

    # Cas spécial pour "Hier"
    if selected_period_label == "Hier":
        end_date_filter = (now_tz - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

elif filter_mode == "Plage de dates personnalisée":
    today_naive = datetime.now().date() # Pour le widget date_input qui est naïf
    start_date_default_naive = (today_naive - timedelta(days=90))
    end_date_default_naive = today_naive

    date_range_custom = st.sidebar.date_input(
        "Sélectionnez la période pour les graphiques :",
        value=(start_date_default_naive, end_date_default_naive),
        max_value=today_naive
    )

    if len(date_range_custom) == 2:
        start_date_filter = tz.localize(datetime.combine(date_range_custom[0], datetime.min.time()))
        end_date_filter = tz.localize(datetime.combine(date_range_custom[1], datetime.max.time()))
    else:
        # Cas où une seule date est sélectionnée (cela peut arriver si l'utilisateur en retire une)
        start_date_filter = tz.localize(datetime.combine(date_range_custom[0], datetime.min.time()))
        end_date_filter = tz.localize(datetime.combine(date_range_custom[0], datetime.max.time()))

# Fallback si, pour une raison improbable, les dates ne sont pas définies
if start_date_filter is None or end_date_filter is None:
    start_date_filter = datetime(2000, 1, 1, tzinfo=tz)
    end_date_filter = now_tz.replace(hour=23, minute=59, second=59, microsecond=999999)

# Affichage de la période sélectionnée (pour confirmation)
mois_fr = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
    7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}

start_date_str = f"{start_date_filter.day} {mois_fr[start_date_filter.month]} {start_date_filter.year}"
end_date_str = f"{end_date_filter.day} {mois_fr[end_date_filter.month]} {end_date_filter.year}"


# --- Titre du Dashboard et CSS ---
st.title("📊 Dashboard des Avis Nickel")
st.markdown("Ce tableau de bord présente une analyse des avis clients collectés pour Nickel.")

st.divider()

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

    .kpi-container-background {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .kpi-positive { color: #2e7d32; font-weight: bold; }
    .kpi-negative { color: #d32f2f; font-weight: bold; }
    .kpi-neutral { color: #f57c00; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# --- CALCUL DES DATAFRAMES UTILISÉS PAR PLUSIEURS SECTIONS (KPIs, Graphiques, Chatbot) ---
# Ceci doit être fait avant les blocs de colonnes pour assurer l'accessibilité globale.

# Calcul pour df_notes_distribution (utilisé par le graphique des notes ET le chatbot)
query_notes_distribution = f"""
SELECT
    note_avis,
    COUNT(id) AS nombre_avis
FROM reviews_nickel
WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}'
GROUP BY note_avis
ORDER BY note_avis ASC;
"""
df_notes_distribution = run_query(query_notes_distribution)

if not df_notes_distribution.empty:
    all_notes = pd.DataFrame({'note_avis': [1, 2, 3, 4, 5]})
    df_notes_distribution = pd.merge(all_notes, df_notes_distribution, on='note_avis', how='left').fillna(0)
    df_notes_distribution['nombre_avis'] = df_notes_distribution['nombre_avis'].astype(int)
    df_notes_distribution = df_notes_distribution.reset_index(drop=True)

# Calcul pour df_evolution (utilisé par le graphique d'évolution ET le chatbot)
delta_jours = (end_date_filter - start_date_filter).days
if delta_jours < 14:
    date_trunc_unit = 'day'
    x_axis_label = 'Date'
else:
    date_trunc_unit = 'week'
    x_axis_label = 'Semaine'

query_evolution = f"""
SELECT
    DATE_TRUNC('{date_trunc_unit}', date_publication) AS periode,
    AVG(note_avis) AS note_moyenne,
    COUNT(id) AS nombre_avis,
    COUNT(CASE WHEN avis_sur_invitation = TRUE THEN 1 END) AS avis_invitation_count,
    COUNT(id) AS total_avis_periode,
    AVG(CASE WHEN avis_sur_invitation = TRUE THEN note_avis END) AS note_moyenne_invitation,
    COUNT(CASE WHEN avis_sur_invitation = TRUE THEN id END) AS nombre_avis_invitation_periode
FROM reviews_nickel
WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}'
GROUP BY periode
ORDER BY periode;
"""
df_evolution = run_query(query_evolution)

if not df_evolution.empty:
    df_evolution = df_evolution.rename(columns={'periode': 'x_axis_data'})
    df_evolution['pourcentage_invitation'] = (df_evolution['avis_invitation_count'] / df_evolution['total_avis_periode']) * 100
    df_evolution = df_evolution.sort_values('x_axis_data')
    df_evolution['x_axis_data'] = pd.to_datetime(df_evolution['x_axis_data']).dt.date
    if date_trunc_unit == 'week':
        df_evolution['x_axis_data_label'] = df_evolution['x_axis_data'].apply(
            lambda d: f"Semaine {d.isocalendar()[1]} ({d.year})"
        )
    else:
        df_evolution['x_axis_data_label'] = df_evolution['x_axis_data'].astype(str)
    df_evolution = df_evolution.reset_index(drop=True)


# Calcul pour df_negative_reviews (utilisé par la section avis négatifs ET le chatbot)
query_negative_reviews = f"""
SELECT
    note_avis,
    contenu_avis,
    date_publication,
    reponse
FROM reviews_nickel
WHERE note_avis <= 2
  AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}'
ORDER BY date_publication DESC;
"""
df_negative_reviews = run_query(query_negative_reviews)


# --- Section: Indicateurs Clés de Performance (KPIs) et Graphiques d'Évolution ---

pcolkpi, gcolkpi = st.columns([1, 2])
with st.container():

    with pcolkpi:

        with st.container(border=True):
            st.header("Indicateurs Clés de Performance")
            st.write(f"**Période sélectionnée :** entre le {start_date_str} et le {end_date_str}")

            colkpi1, colkpi2 = st.columns(2)

            # KPI 1: Nombre d'avis reçus
            query_kpi_avis_periode = f"""
            SELECT COUNT(id)
            FROM reviews_nickel
            WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
            """
            df_avis_periode = run_query(query_kpi_avis_periode)
            nombre_avis_periode = df_avis_periode.iloc[0, 0] if not df_avis_periode.empty else 0
            colkpi1.metric("Nombre d'avis reçus", nombre_avis_periode)

            # KPI 2: Pourcentage d'avis répondus
            query_kpi_reponse = f"""
            SELECT
                COUNT(CASE WHEN reponse = TRUE THEN 1 END) AS count_responded,
                COUNT(id) AS total_reviews
            FROM reviews_nickel
            WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
            """
            df_reponse = run_query(query_kpi_reponse)
            pourcentage_reponse = 0
            if not df_reponse.empty and df_reponse.iloc[0]['total_reviews'] > 0:
                pourcentage_reponse = (df_reponse.iloc[0]['count_responded'] / df_reponse.iloc[0]['total_reviews']) * 100
            colkpi1.metric("% Avis répondus", f"{pourcentage_reponse:.1f}%")

            # KPI 3: Temps de réponse moyen
            query_kpi_temps_reponse = f"""
            SELECT AVG(EXTRACT(EPOCH FROM (date_reponse - date_publication))) AS avg_response_seconds
            FROM reviews_nickel
            WHERE reponse = TRUE
            AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}'
            AND date_reponse > date_publication;
            """
            df_temps_reponse = run_query(query_kpi_temps_reponse)
            temps_moyen_secondes = df_temps_reponse.iloc[0, 0] if not df_temps_reponse.empty and df_temps_reponse.iloc[0, 0] is not None else 0
            temps_moyen_td = timedelta(seconds=float(temps_moyen_secondes))
            colkpi2.metric("Temps de réponse moyen", format_timedelta(temps_moyen_td))

            # KPI 4: Note moyenne des avis
            query_kpi_note_periode = f"""
            SELECT AVG(note_avis)
            FROM reviews_nickel
            WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
            """
            df_note_periode = run_query(query_kpi_note_periode)
            note_moyenne_periode = df_note_periode.iloc[0, 0] if not df_note_periode.empty and df_note_periode.iloc[0, 0] is not None else 0
            colkpi2.metric("Note moyenne", f"{note_moyenne_periode:.2f}/5")

 
    with gcolkpi:

        # --- Graphique Combiné ---
        # Cette section utilise maintenant le df_evolution pré-calculé
        st.subheader("Évolution Combinée : Nombre d'avis et Note moyenne")

        if not df_evolution.empty:
            df_evolution['nombre_avis'] = pd.to_numeric(df_evolution['nombre_avis'], errors='coerce').astype('int64')
            df_evolution.dropna(subset=['nombre_avis'], inplace=True)

            fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

            x_labels_list = df_evolution['x_axis_data_label'].tolist()
            nombre_avis_list = df_evolution['nombre_avis'].tolist()
            note_moyenne_list = df_evolution['note_moyenne'].tolist()

            # Ajout du Nombre d'avis en barres (axe Y1)
            fig_combined.add_trace(go.Bar(
                x=x_labels_list,
                y=nombre_avis_list,
                name='Nombre d\'avis',
                marker_color='#5B9BD5',
                orientation='v',
                yaxis='y'
            ), secondary_y=False)

            # Ajout de la Note moyenne en ligne (axe Y2)
            fig_combined.add_trace(go.Scatter(
                x=x_labels_list,
                y=note_moyenne_list,
                mode='lines+markers',
                name='Note moyenne',
                line=dict(color='#ED7D31', width=3),
                marker=dict(symbol='circle', size=8, color='#ED7D31'),
                yaxis='y2'
            ), secondary_y=True)

            # Configuration des axes et du layout
            fig_combined.update_layout(
                xaxis=dict(type='category', title=x_axis_label),
                yaxis=dict(
                    title='Nombre d\'avis',
                    title_font=dict(color='#5B9BD5'),
                    tickfont=dict(color='#5B9BD5'),
                    range=[0, df_evolution['nombre_avis'].max() * 1.1]
                ),
                yaxis2=dict(
                    title='Note moyenne',
                    title_font=dict(color='#ED7D31'),
                    tickfont=dict(color='#ED7D31'),
                    range=[1, 5]
                ),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(255,255,255,0)'),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                height=450
            )

            st.plotly_chart(fig_combined, use_container_width=True)

        else:
            st.info("Pas de données disponibles pour l'évolution combinée sur la période sélectionnée.")

st.divider()

################################################################
################################################################

# --- Section: Suivi des avis sur invitation ---
gcol_asi, pcol_asi = st.columns([2, 1])

with pcol_asi:

    with st.container(border=True):
        st.header("Suivi des avis sur invitation")
        st.markdown("Cette section analyse spécifiquement les avis reçus suite à une invitation.")
        pcol_asi1, pcol_asi2 = st.columns(2)

        # KPI 1: Nombre d'avis sur invitation reçus
        query_kpi_invit_count = f"""
        SELECT COUNT(id)
        FROM reviews_nickel
        WHERE avis_sur_invitation = TRUE
        AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        """
        df_invit_count = run_query(query_kpi_invit_count)
        nombre_avis_invit = df_invit_count.iloc[0, 0] if not df_invit_count.empty else 0
        pcol_asi2.metric("Avis sur invitation reçus", nombre_avis_invit)

        # KPI 2: Note moyenne avis sur invitation
        query_kpi_invit_note = f"""
        SELECT AVG(note_avis)
        FROM reviews_nickel
        WHERE avis_sur_invitation = TRUE
        AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        """
        df_invit_note = run_query(query_kpi_invit_note)
        note_moyenne_invit = df_invit_note.iloc[0, 0] if not df_invit_note.empty and df_invit_note.iloc[0, 0] is not None else np.nan
        pcol_asi2.metric("Note moyenne avis sur invitation", f"{note_moyenne_invit:.2f}/5" if not pd.isna(note_moyenne_invit) else "N/A")

        # KPI 3: Pourcentage d'avis sur invitation (parmi tous les avis)
        query_kpi_pourcentage_invitation = f"""
        SELECT
            COUNT(CASE WHEN avis_sur_invitation = TRUE THEN 1 END) AS count_invited,
            COUNT(id) AS total_reviews
        FROM reviews_nickel
        WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        """
        df_pourcentage_invitation = run_query(query_kpi_pourcentage_invitation)
        pourcentage_avis_invitation = 0
        if not df_pourcentage_invitation.empty and df_pourcentage_invitation.iloc[0]['total_reviews'] > 0:
            pourcentage_avis_invitation = (df_pourcentage_invitation.iloc[0]['count_invited'] / df_pourcentage_invitation.iloc[0]['total_reviews']) * 100
        pcol_asi1.metric("% Avis sur invitation", f"{pourcentage_avis_invitation:.1f}%")

        # KPI 4: Différence de note (avis avec invitation vs sans invitation)
        query_note_moyenne_sans_invitation = f"""
        SELECT AVG(note_avis)
        FROM reviews_nickel
        WHERE avis_sur_invitation = FALSE
        AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        """
        df_note_sans_invitation = run_query(query_note_moyenne_sans_invitation)
        note_moyenne_sans_invitation_kpi = df_note_sans_invitation.iloc[0, 0] if not df_note_sans_invitation.empty and df_note_sans_invitation.iloc[0, 0] is not None else np.nan

        valeur_principale_diff_invit_vs_sans_invit = np.nan
        valeur_principale_str = "N/A"
        if not pd.isna(note_moyenne_invit) and not pd.isna(note_moyenne_sans_invitation_kpi):
            valeur_principale_diff_invit_vs_sans_invit = note_moyenne_invit - note_moyenne_sans_invitation_kpi
            valeur_principale_str = f"{valeur_principale_diff_invit_vs_sans_invit:+.2f}"

        pcol_asi1.metric(
            "Diff. Note (Invit. vs Sans Invit.)",
            valeur_principale_str,
        )

    # GRAPHIQUE COMBINÉ : Évolution du nombre d'avis sur invitation et de la note moyenne
with gcol_asi:
    st.subheader("Évolution Combinée : Avis sur invitation et Note moyenne")
    # Cette section utilise maintenant le df_evolution pré-calculé
    if not df_evolution.empty:
        df_evolution['nombre_avis_invitation_periode'] = pd.to_numeric(df_evolution['nombre_avis_invitation_periode'], errors='coerce').astype('int64')
        df_evolution['note_moyenne_invitation'] = pd.to_numeric(df_evolution['note_moyenne_invitation'], errors='coerce')
        df_evolution_invit_filtered = df_evolution.dropna(subset=['nombre_avis_invitation_periode', 'note_moyenne_invitation'])

        if not df_evolution_invit_filtered.empty:
            fig_combined_invit = make_subplots(specs=[[{"secondary_y": True}]])

            x_labels_list_invit = df_evolution_invit_filtered['x_axis_data_label'].tolist()
            nombre_avis_invit_list = df_evolution_invit_filtered['nombre_avis_invitation_periode'].tolist()
            note_moyenne_invit_list = df_evolution_invit_filtered['note_moyenne_invitation'].tolist()

            # Ajout du Nombre d'avis sur invitation en barres (axe Y1)
            fig_combined_invit.add_trace(go.Bar(
                x=x_labels_list_invit,
                y=nombre_avis_invit_list,
                name='Nombre d\'avis sur invitation',
                marker_color='#FF9933',
                orientation='v',
                yaxis='y'
            ), secondary_y=False)

            # Ajout de la Note moyenne sur invitation en ligne (axe Y2)
            fig_combined_invit.add_trace(go.Scatter(
                x=x_labels_list_invit,
                y=note_moyenne_invit_list,
                mode='lines+markers',
                name='Note moyenne (invitation)',
                line=dict(color='#00CC99', width=3),
                marker=dict(symbol='circle', size=8, color='#00CC99'),
                yaxis='y2'
            ), secondary_y=True)

            # Configuration des axes et du layout pour le graphique avis invitation
            fig_combined_invit.update_layout(
                xaxis=dict(type='category', title=x_axis_label),
                yaxis=dict(
                    title='Nombre d\'avis sur invitation',
                    title_font=dict(color='#FF9933'),
                    tickfont=dict(color='#FF9933'),
                    range=[0, df_evolution_invit_filtered['nombre_avis_invitation_periode'].max() * 1.1]
                ),
                yaxis2=dict(
                    title='Note moyenne (invitation)',
                    title_font=dict(color='#00CC99'),
                    tickfont=dict(color='#00CC99'),
                    range=[1, 5]
                ),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(255,255,255,0)'),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                height=450
            )

            st.plotly_chart(fig_combined_invit, use_container_width=True)
        else:
            st.info("Pas de données d'avis sur invitation disponibles pour la période sélectionnée.")
    else:
        st.info("Pas de données disponibles pour l'évolution combinée des avis sur invitation sur la période sélectionnée.")

st.divider()

#################################################################
#################################################################

# --- Section: Étude des avis négatifs ---
st.header("Étude des avis négatifs")
with st.container(border=True):
    st.markdown("Cette section présente les avis avec une note de 1 ou 2, considérés comme négatifs.")

    # Cette section utilise maintenant le df_negative_reviews pré-calculé
    if not df_negative_reviews.empty:
        df_negative_reviews['reponse_texte'] = df_negative_reviews['reponse'].apply(lambda x: 'Oui' if x else 'Non')

        def style_response(val):
            color = 'green' if val == 'Oui' else 'red'
            return f'color: {color}; font-weight: bold;'

        df_display = df_negative_reviews[['note_avis', 'contenu_avis', 'date_publication', 'reponse_texte']].copy()
        df_display.columns = ['Note', 'Contenu de l\'avis', 'Date de publication', 'Réponse apportée']

        styled_df_negative = df_display.style.applymap(style_response, subset=['Réponse apportée'])

        st.dataframe(styled_df_negative, use_container_width=True)
    else:
        st.info("Aucun avis négatif trouvé pour la période sélectionnée.")

###############################################################
# --- Chatbot Assistant : Génération automatique du rapport ---

st.divider()
st.subheader("🤖 Rapport Automatique")

with st.spinner("Génération du rapport..."):
    # voir si possible de ne pas recalculer les KPIs si déjà fait

    # KPI spécifiques pour le rapport
    query_kpi_avis_periode = f"""
    SELECT COUNT(id)
    FROM reviews_nickel
    WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    nombre_avis_periode_report = run_query(query_kpi_avis_periode).iloc[0, 0] if not run_query(query_kpi_avis_periode).empty else 0

    query_kpi_reponse = f"""
    SELECT
        COUNT(CASE WHEN reponse = TRUE THEN 1 END) AS count_responded,
        COUNT(id) AS total_reviews
    FROM reviews_nickel
    WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    df_reponse_report = run_query(query_kpi_reponse)
    pourcentage_reponse_report = (df_reponse_report.iloc[0]['count_responded'] / df_reponse_report.iloc[0]['total_reviews']) * 100 if not df_reponse_report.empty and df_reponse_report.iloc[0]['total_reviews'] > 0 else 0

    query_kpi_temps_reponse = f"""
    SELECT AVG(EXTRACT(EPOCH FROM (date_reponse - date_publication))) AS avg_response_seconds
    FROM reviews_nickel
    WHERE reponse = TRUE
    AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}'
    AND date_reponse > date_publication;
    """
    temps_moyen_secondes_report = run_query(query_kpi_temps_reponse).iloc[0, 0] if not run_query(query_kpi_temps_reponse).empty and run_query(query_kpi_temps_reponse).iloc[0, 0] is not None else 0
    temps_moyen_td_report = timedelta(seconds=float(temps_moyen_secondes_report))

    query_kpi_note_periode = f"""
    SELECT AVG(note_avis)
    FROM reviews_nickel
    WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    note_moyenne_periode_report = run_query(query_kpi_note_periode).iloc[0, 0] if not run_query(query_kpi_note_periode).empty and run_query(query_kpi_note_periode).iloc[0, 0] is not None else 0

    # Données spécifiques aux avis sur invitation (utilisées précédemment dans le dashboard)
    query_kpi_invit_count = f"""
    SELECT COUNT(id)
    FROM reviews_nickel
    WHERE avis_sur_invitation = TRUE
    AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    nombre_avis_invit_report = run_query(query_kpi_invit_count).iloc[0, 0] if not run_query(query_kpi_invit_count).empty else 0

    query_kpi_invit_note = f"""
    SELECT AVG(note_avis)
    FROM reviews_nickel
    WHERE avis_sur_invitation = TRUE
    AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    note_moyenne_invit_report = run_query(query_kpi_invit_note).iloc[0, 0] if not run_query(query_kpi_invit_note).empty and run_query(query_kpi_invit_note).iloc[0, 0] is not None else np.nan

    query_kpi_pourcentage_invitation = f"""
    SELECT
        COUNT(CASE WHEN avis_sur_invitation = TRUE THEN 1 END) AS count_invited,
        COUNT(id) AS total_reviews
    FROM reviews_nickel
    WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    df_pourcentage_invitation_report = run_query(query_kpi_pourcentage_invitation)
    pourcentage_avis_invitation_report = (df_pourcentage_invitation_report.iloc[0]['count_invited'] / df_pourcentage_invitation_report.iloc[0]['total_reviews']) * 100 if not df_pourcentage_invitation_report.empty and df_pourcentage_invitation_report.iloc[0]['total_reviews'] > 0 else 0

    query_note_moyenne_sans_invitation = f"""
    SELECT AVG(note_avis)
    FROM reviews_nickel
    WHERE avis_sur_invitation = FALSE
    AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
    """
    note_moyenne_sans_invitation_report = run_query(query_note_moyenne_sans_invitation).iloc[0, 0] if not run_query(query_note_moyenne_sans_invitation).empty and run_query(query_note_moyenne_sans_invitation).iloc[0, 0] is not None else np.nan

    valeur_principale_diff_invit_vs_sans_invit_report = np.nan
    if not pd.isna(note_moyenne_invit_report) and not pd.isna(note_moyenne_sans_invitation_report):
        valeur_principale_diff_invit_vs_sans_invit_report = note_moyenne_invit_report - note_moyenne_sans_invitation_report

    kpi_summary = f"""
    Période analysée: du {start_date_str} au {end_date_str}.
    Nombre total d'avis: {nombre_avis_periode_report}.
    Note moyenne générale: {note_moyenne_periode_report:.2f}/5.
    Pourcentage d'avis répondus: {pourcentage_reponse_report:.1f}%.
    Temps de réponse moyen: {format_timedelta(temps_moyen_td_report)}.

    Avis sur invitation:
      - Nombre d'avis sur invitation: {nombre_avis_invit_report}.
      - Note moyenne des avis sur invitation: {note_moyenne_invit_report:.2f}/5.
      - Pourcentage d'avis sur invitation (parmi tous les avis): {pourcentage_avis_invitation_report:.1f}%.
      - Différence de note (invités vs non-invités): {valeur_principale_diff_invit_vs_sans_invit_report:+.2f} points (si applicable).
    """

    notes_dist_summary = "Répartition des notes:\n"
    if not df_notes_distribution.empty:
        for index, row in df_notes_distribution.iterrows():
            notes_dist_summary += f"  Note {int(row['note_avis'])}: {int(row['nombre_avis'])} avis.\n"
    else:
        notes_dist_summary += "  Aucune donnée de répartition des notes disponible pour cette période.\n"

    negative_reviews_summary = "Avis négatifs (note <= 2):\n"
    if not df_negative_reviews.empty:
        # Limiter à quelques exemples pour ne pas surcharger le prompt
        for index, row in df_negative_reviews.head(3).iterrows():
            content_preview = row['contenu_avis'][:150] + "..." if len(row['contenu_avis']) > 150 else row['contenu_avis']
            negative_reviews_summary += f"- Note {row['note_avis']}: \"{content_preview}\" (Répondu: {'Oui' if row['reponse'] else 'Non'}).\n"
    else:
        negative_reviews_summary += "  Aucun avis négatif pour cette période.\n"

    full_context_for_llm = f"""
    Contexte: Vous êtes un expert en analyse d'avis clients pour la marque Nickel.
    Votre tâche est de générer un mini-rapport clair, concis et actionnable (3-4 paragraphes)
    basé sur les données d'avis clients fournies pour la période spécifiée.
    Mettez en évidence les tendances positives et négatives, les chiffres clés, les observations concernant les avis sur invitation, et formulez des observations pertinentes.

    Voici les données pour le rapport:
    {kpi_summary}
    {notes_dist_summary}
    {negative_reviews_summary}

    Générez le rapport en vous basant uniquement sur ces données.
    """

    # 2. Appel à l'API du LLM (Exemple avec un placeholder)
    # Pour utiliser un vrai LLM comme OpenAI ou Gemini, vous devrez :
    # 1. Installer la bibliothèque (ex: pip install openai google-generativeai)
    # 2. Configurer votre clé API (st.secrets["OPENAI_API_KEY"] ou st.secrets["GEMINI_API_KEY"])
    # 3. Décommenter et adapter le code d'appel à l'API.

    chatbot_response = f"**Rapport d'analyse des avis clients pour Nickel**\n\n"
    chatbot_response += f"Sur la période du **{start_date_str} au {end_date_str}**, nous avons collecté **{nombre_avis_periode_report}** avis, avec une note moyenne de **{note_moyenne_periode_report:.2f}/5**.\n"
    chatbot_response += f"Le taux de réponse aux avis s'élève à **{pourcentage_reponse_report:.1f}%**, avec un temps de réponse moyen de **{format_timedelta(temps_moyen_td_report)}**.\n\n"
    chatbot_response += f"**Focus sur les avis sur invitation :**\n"
    chatbot_response += f"  - **{nombre_avis_invit_report}** avis ont été reçus suite à une invitation, représentant **{pourcentage_avis_invitation_report:.1f}%** du total des avis.\n"
    if not pd.isna(note_moyenne_invit_report):
        chatbot_response += f"  - La note moyenne pour ces avis invités est de **{note_moyenne_invit_report:.2f}/5**.\n"
    if not pd.isna(valeur_principale_diff_invit_vs_sans_invit_report):
        chatbot_response += f"  - La différence de note entre les avis invités et non invités est de **{valeur_principale_diff_invit_vs_sans_invit_report:+.2f}** points.\n\n"
    else:
        chatbot_response += "  - Il n'y a pas assez de données pour calculer la note moyenne ou la différence avec les avis non-invités.\n\n"

    chatbot_response += f"La répartition des notes est la suivante :\n{notes_dist_summary}\n"
    chatbot_response += f"**Points d'attention** : {negative_reviews_summary}"


    st.markdown(chatbot_response)