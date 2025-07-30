import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np # Pour générer des données d'exemple

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Dashboard Avis Nickel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Exemple de données (pour rendre les graphiques fonctionnels) ---
dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
data1 = np.random.rand(30) * 100
data2 = np.random.rand(30) * 5 + 1 # Notes de 1 à 5
df_example = pd.DataFrame({'Date': dates, 'Valeur1': data1, 'Valeur2': data2})

# --- Création du premier graphique (le plus petit, 1/3) ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_example['Date'], y=df_example['Valeur1'], mode='lines', name='Graphique 1'))
fig1.update_layout(title_text='Graphique 1 (1/3 Largeur)', height=300) # Hauteur ajustée pour la démo

# --- Création du deuxième graphique (le plus grand, 2/3) ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_example['Date'], y=df_example['Valeur2'], mode='lines+markers', name='Graphique 2'))
fig2.update_layout(title_text='Graphique 2 (2/3 Largeur)', height=300) # Hauteur ajustée pour la démo
fig2.update_yaxes(range=[1, 5])

# --- Utilisation de st.columns pour la mise en page ---

# Créer deux colonnes. La première aura une largeur relative de 1, la seconde de 2.
# Cela signifie que la première prendra 1/(1+2) = 1/3 de la largeur
# et la seconde prendra 2/(1+2) = 2/3 de la largeur.
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Section 1 (1/3)")
    st.plotly_chart(fig1, use_container_width=True) # use_container_width=True est important

with col2:
    st.header("Section 2 (2/3)")
    st.plotly_chart(fig2, use_container_width=True) # use_container_width=True est important

st.write("---")
st.write("Vous pouvez voir que le Graphique 2 est deux fois plus large que le Graphique 1.")

# --- Exemple avec trois graphiques si besoin de plus de contrôle ---
st.header("Exemple avec trois colonnes (pour plus de flexibilité)")
col_a, col_b, col_c = st.columns([0.5, 1, 1.5]) # Largeurs relatives différentes

with col_a:
    st.subheader("Très petit")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Moyen")
    st.plotly_chart(fig2, use_container_width=True)

with col_c:
    st.subheader("Grand")
    st.plotly_chart(fig1, use_container_width=True) # Réutilisation de fig1 pour la démo