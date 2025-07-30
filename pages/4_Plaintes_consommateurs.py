import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# Titre de l'application
st.title("😡 Analyse des Plaintes Consommateurs")
st.write("\n")
st.write("\n")
st.write("\n")


# Chargement des données
@st.cache_data
def load_data():
    file_path = Path("data") / "plaintes_consommateurs.xlsx"
    return pd.read_excel(file_path, sheet_name="Data")


df = load_data()
st.session_state.df = df
df = st.session_state.df

# Préparation
df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
df["Year"] = df["Date received"].dt.year


# Slider de sélection d'année
years = sorted(df["Year"].dropna().unique())
selected_year = st.sidebar.slider(
    "Choisir une année", int(min(years)), int(max(years)), int(max(years))
)
# Filtre produit
product_options = ["Tous"] + sorted(df["Product"].dropna().unique())
selected_product = st.sidebar.selectbox("Filtrer par produit", product_options)

# Filtre multi-états
state_options = sorted(df["State"].dropna().unique())
selected_states = st.sidebar.multiselect("Filtrer par État(s)", state_options)

# Filtre par délai de réponse
timely_filter = st.sidebar.checkbox(
    "Afficher uniquement les réponses non traitées dans les délais (No)"
)

# Filtrage des données global
filteredglobal_df = len(df)
# Filtrage des données selon l'année sélectionnée
filtered_df = df[df["Year"] == selected_year]

# Filtre produit
if selected_product != "Tous":
    filtered_df = filtered_df[filtered_df["Product"] == selected_product]

# Filtre multi-états
if selected_states:
    filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

# Filtre délai de réponse
if timely_filter:
    filtered_df = filtered_df[filtered_df["Timely response"] == "No"]


st.write("\n")
st.write("\n")

st.markdown(
    f"<h4>📅 <span style='color:#db55ff'> Année sélectionnée :</span> <span style='color:#88f572'>{selected_year}</span></h4>",
    unsafe_allow_html=True,
)

st.divider()

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Affichage du nombre total de plaintes global
    st.markdown(
        f"<h6 style='color:#ab0093;'>Nombre total de plaintes </h6>",
        unsafe_allow_html=True,
    )
    st.metric(label=" ", value=f"{filteredglobal_df:,}")

with col2:
    # Affichage du nombre total de plaintes global
    st.markdown(
        f"<h6 style='color:#ab0093;'> Nombre de plaintes en {selected_year}</h6>",
        unsafe_allow_html=True,
    )
    st.metric(
        label=" ",
        value=f"{filtered_df.shape[0]:,}",
    )

with col3:
    # Graphique 1 : Nombre de plaintes par année
    complaints_by_year = df["Year"].value_counts().sort_index()
    fig1 = px.line(
        x=complaints_by_year.index,
        y=complaints_by_year.values,
        labels={"x": " ", "y": " "},
    )
    fig1.update_layout(
        title={
            "text": "Évolution du nombre de plaintes",
            "font": {"color": "#ab0093", "size": 15},
        }
    )
    fig1.update_traces(line=dict(color="#ff7e04", width=2))

    st.plotly_chart(fig1)

col4, col5 = st.columns([2, 1])

with col4:
    # Graphique 2 : Top 10 des produits concernés
    top_products = filtered_df["Product"].value_counts().nlargest(10)
    # ➤ 2. Création du graphique avec Plotly Express
    fig2 = px.bar(
        x=top_products.index,
        y=top_products.values,
        labels={"x": "", "y": ""},
    )

    # ➤ 3. Personnalisation de l'apparence
    fig2.update_layout(
        title={
            "text": "Top 10 des plaintes les plus fréquentes",
            "font": {"color": "#ab0093", "size": 15},
        },
        xaxis_tickangle=-20,  # Incliner les labels si trop longs
        xaxis_tickfont=dict(size=8),
    )

    fig2.update_traces(marker_color="#ff922d")
    # ➤ 4. Affichage dans Streamlit
    st.plotly_chart(fig2)
with col5:
    # Graphique 3 : Taux de réponse dans les délais
    timely = filtered_df["Timely response"].value_counts(normalize=True) * 100

    couleurs = {"Yes": "#ab0093", "No": "#ff922d"}  # vert  # rouge

    fig4 = px.pie(
        names=timely.index,
        values=timely.values,
        color=timely.index,  # important pour appliquer les couleurs
        color_discrete_map=couleurs,  # dictionnaire de couleurs
    )
    fig4.update_layout(
        title={
            "text": "Délai de réponse respecté",
            "font": {"color": "#ab0093", "size": 15},
        }
    )
    st.plotly_chart(fig4)


# ➤ Calcul des plaintes par État
state_counts = filtered_df["State"].value_counts().reset_index()
state_counts.columns = ["State", "Nombre de plaintes"]

# ➤ Création de la carte interactive
fig3 = px.choropleth(
    state_counts,
    locations="State",  # colonne des codes d’État (ex: "CA")
    locationmode="USA-states",  # mode : États-Unis par abréviation
    color="Nombre de plaintes",  # colonne à colorier
    color_continuous_scale="Turbo",  # palette de couleurs
    scope="usa",  # centrage sur les USA
    labels={"Nombre de plaintes": "Plaintes"},
    title="États avec le plus de plaintes",
)

# ➤ Mise en forme optionnelle
fig3.update_layout(
    title_font=dict(size=15, color="#ab0093"),
    geo=dict(bgcolor="rgba(0,0,0,0)"),  # fond transparent
    margin=dict(l=0, r=0, t=150, b=0),
    height=500,  # Hauteur en pixels
    width=1000,  # Largeur en pixels (utile hors Streamlit)
)

# ➤ Affichage Streamlit
st.plotly_chart(fig3, use_container_width=True)


df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
df["Year"] = df["Date received"].dt.year


filtered_df = df[df["Year"] == selected_year]
top_states = filtered_df["State"].value_counts().nlargest(10)


fig5 = px.bar(
    x=top_states.index,
    y=top_states.values,
    labels={"x": "", "y": ""},
    title=" Top 10 des États avec le plus de plaintes",
)
fig5.update_layout(
    title_font=dict(size=15, color="#ab0093"),
    geo=dict(bgcolor="rgba(0,0,0,0)"),  # fond transparent
    margin=dict(l=0, r=0, t=150, b=0),
    height=500,  # Hauteur en pixels
    width=800,  # Largeur en pixels (utile hors Streamlit)
)

fig5.update_traces(marker_color="#ff922d")

st.plotly_chart(fig5, use_container_width=True)


# st.dataframe(df)
