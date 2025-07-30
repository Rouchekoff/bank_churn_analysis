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

today = datetime.now()
start_date_default = today - timedelta(days=90)
end_date_default = today

date_range = st.sidebar.date_input(
    "Sélectionnez la période pour les graphiques :",
    value=(start_date_default, end_date_default),
    max_value=today
)

if len(date_range) == 2:
    start_date_filter = datetime.combine(date_range[0], datetime.min.time())
    end_date_filter = datetime.combine(date_range[1], datetime.max.time())
else:
    start_date_filter = datetime.combine(date_range[0], datetime.min.time())
    end_date_filter = datetime.combine(date_range[0], datetime.max.time())

tz = pytz.timezone("Europe/Paris")
now_tz = tz.localize(datetime.now())

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


# --- Section: Indicateurs Clés de Performance (KPIs) et Graphiques d'Évolution ---

pcolkpi, gcolkpi = st.columns([1, 2])
with st.container():

    with pcolkpi:
        
        with st.container(border=True):
            st.header("Indicateurs Clés de Performance")
            st.write(f"**Période sélectionnée :** entre le {start_date_str} et le {end_date_str}")

            colkpi1, colkpi2 = st.columns(2)

            query_kpi_avis_periode = f"""
            SELECT COUNT(id)
            FROM reviews_nickel
            WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
            """
            df_avis_periode = run_query(query_kpi_avis_periode)
            nombre_avis_periode = df_avis_periode.iloc[0, 0] if not df_avis_periode.empty else 0
            colkpi1.metric("Nombre d'avis reçus", nombre_avis_periode)

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

            query_kpi_note_periode = f"""
            SELECT AVG(note_avis)
            FROM reviews_nickel
            WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
            """
            df_note_periode = run_query(query_kpi_note_periode)
            note_moyenne_periode = df_note_periode.iloc[0, 0] if not df_note_periode.empty and df_note_periode.iloc[0, 0] is not None else 0
            colkpi2.metric("Note moyenne", f"{note_moyenne_periode:.2f}/5")

    with gcolkpi:
            
        # Logique de granularité pour les graphiques d'évolution
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

            # *** C'EST LA NOUVELLE LIGNE CLÉ À AJOUTER ***
            df_evolution = df_evolution.reset_index(drop=True)
            # *** FIN DE LA NOUVELLE LIGNE CLÉ ***

        # --- Graphique Combiné ---
        st.subheader("Évolution Combinée : Nombre d'avis et Note moyenne")

        if not df_evolution.empty:
            # Assurer que nombre_avis est bien numérique et entier
            df_evolution['nombre_avis'] = pd.to_numeric(df_evolution['nombre_avis'], errors='coerce').astype('int64')
            df_evolution.dropna(subset=['nombre_avis'], inplace=True)

            fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

            # Convertir en listes pures au cas où (dernière tentative de ce côté)
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
        # 4 colonnes pour les KPIs d'invitation
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

        
        # NOUVEAU KPI 4: Différence de note (avis avec invitation vs sans invitation) et son delta

        # Récupération de la note moyenne des avis sur invitation (déjà calculée plus haut)
        # note_moyenne_invit est déjà disponible depuis col_invit_kpi2

        # Récupération de la note moyenne des avis sans invitation
        query_note_moyenne_sans_invitation = f"""
        SELECT AVG(note_avis)
        FROM reviews_nickel
        WHERE avis_sur_invitation = FALSE
        AND date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        """
        df_note_sans_invitation = run_query(query_note_moyenne_sans_invitation)
        note_moyenne_sans_invitation_kpi = df_note_sans_invitation.iloc[0, 0] if not df_note_sans_invitation.empty and df_note_sans_invitation.iloc[0, 0] is not None else np.nan

        # Note moyenne globale (déjà calculée au début dans 'note_moyenne_periode')
        # Cette variable doit être accessible ici. Si elle ne l'est pas, décommentez le bloc ci-dessous:
        # query_kpi_note_periode = f"""
        # SELECT AVG(note_avis)
        # FROM reviews_nickel
        # WHERE date_publication >= '{start_date_filter.isoformat()}' AND date_publication <= '{end_date_filter.isoformat()}';
        # """
        # df_note_periode = run_query(query_kpi_note_periode)
        # note_moyenne_periode = df_note_periode.iloc[0, 0] if not df_note_periode.empty and df_note_periode.iloc[0, 0] is not None else np.nan


        # Calcul de la VALEUR PRINCIPALE du KPI : (Note avec invitation) - (Note sans invitation)
        valeur_principale_diff_invit_vs_sans_invit = np.nan
        valeur_principale_str = "N/A"
        if not pd.isna(note_moyenne_invit) and not pd.isna(note_moyenne_sans_invitation_kpi):
            valeur_principale_diff_invit_vs_sans_invit = note_moyenne_invit - note_moyenne_sans_invitation_kpi
            valeur_principale_str = f"{valeur_principale_diff_invit_vs_sans_invit:+.2f}"

        # # Calcul du DELTA : (Note globale) - (Note sans invitation)
        # # Le but du delta est de savoir combien les invitations nous font gagner de points.
        # # C'est la différence entre la note "naturelle" (sans invitation) et la note "boostée" (globale).
        # delta_gain_invitations = np.nan
        # delta_str = "(N/A)"
        # delta_color = "off" # Default to neutral

        # if not pd.isna(note_moyenne_periode) and not pd.isna(note_moyenne_sans_invitation_kpi):
        #     delta_gain_invitations = note_moyenne_periode - note_moyenne_sans_invitation_kpi
        #     delta_str = f"{delta_gain_invitations:+.2f}"

        #     # Définir la couleur du delta
        #     # Un delta positif signifie que les invitations ont "gagné" des points par rapport à la note naturelle (sans invitation)
        #     if delta_gain_invitations > 0.05:
        #         delta_color = "inverse" # Vert (c'est un gain, donc positif)
        #     elif delta_gain_invitations < -0.05:
        #         delta_color = "off" # Rouge (si, étonnamment, les invitations faisaient baisser la note globale)
        #     else:
        #         delta_color = "off" # Neutre (gris) si la différence est minime

        pcol_asi1.metric(
            "Diff. Note (Invit. vs Sans Invit.)", # Titre du KPI
            valeur_principale_str, # Affiche la différence entre invit et sans invit
        #    delta=delta_str, # Affiche le gain/perte des invitations
        #    delta_color=delta_color
        )

    # GRAPHIQUE COMBINÉ : Évolution du nombre d'avis sur invitation et de la note moyenne
with gcol_asi:
    st.subheader("Évolution Combinée : Avis sur invitation et Note moyenne")
    if not df_evolution.empty:
        # Assurer que les colonnes sont numériques et sans NaN pour le graphique
        df_evolution['nombre_avis_invitation_periode'] = pd.to_numeric(df_evolution['nombre_avis_invitation_periode'], errors='coerce').astype('int64')
        df_evolution['note_moyenne_invitation'] = pd.to_numeric(df_evolution['note_moyenne_invitation'], errors='coerce')
        df_evolution_invit_filtered = df_evolution.dropna(subset=['nombre_avis_invitation_periode', 'note_moyenne_invitation'])

        if not df_evolution_invit_filtered.empty:
            fig_combined_invit = make_subplots(specs=[[{"secondary_y": True}]])

            # Convertir en listes pures
            x_labels_list_invit = df_evolution_invit_filtered['x_axis_data_label'].tolist()
            nombre_avis_invit_list = df_evolution_invit_filtered['nombre_avis_invitation_periode'].tolist()
            note_moyenne_invit_list = df_evolution_invit_filtered['note_moyenne_invitation'].tolist()

            # Ajout du Nombre d'avis sur invitation en barres (axe Y1)
            fig_combined_invit.add_trace(go.Bar(
                x=x_labels_list_invit,
                y=nombre_avis_invit_list,
                name='Nombre d\'avis sur invitation',
                marker_color='#FF9933', # Une couleur différente pour les avis invitation
                orientation='v',
                yaxis='y'
            ), secondary_y=False)

            # Ajout de la Note moyenne sur invitation en ligne (axe Y2)
            fig_combined_invit.add_trace(go.Scatter(
                x=x_labels_list_invit,
                y=note_moyenne_invit_list,
                mode='lines+markers',
                name='Note moyenne (invitation)',
                line=dict(color='#00CC99', width=3), # Une autre couleur pour la ligne
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

# --- Section: Étude des avis négatifs (reste inchangée) ---
st.header("Étude des avis négatifs")
with st.container(border=True):
    st.markdown("Cette section présente les avis avec une note de 1 ou 2, considérés comme négatifs.")

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

    st.markdown("---")
    st.caption("Données mises à jour toutes les 10 minutes.")