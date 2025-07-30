import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Tendances et KPIs", page_icon="📈", layout="wide")

# CSS personnalisé
st.markdown(
    """
<style>
    .trend-up { color: #2e7d32; font-weight: bold; }
    .trend-down { color: #d32f2f; font-weight: bold; }
    .trend-stable { color: #f57c00; font-weight: bold; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-evolution {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
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
        avis_df.columns = avis_df.columns.str.strip()

        if "date" in avis_df.columns:
            avis_df["date"] = pd.to_datetime(avis_df["date"], errors="coerce")

        return avis_df
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None


def calculate_trends(avis_df, period="M"):
    """Calcule les tendances temporelles"""
    if "date" not in avis_df.columns:
        return None

    # Grouper par période
    avis_df["period"] = avis_df["date"].dt.to_period(period)

    # Métriques par période
    trends = (
        avis_df.groupby("period")
        .agg(
            {
                "sentiment": lambda x: (
                    (x == "positif").sum() if "sentiment" in avis_df.columns else 0
                ),
                "note": "mean" if "note" in avis_df.columns else lambda x: 0,
            }
        )
        .reset_index()
    )

    trends["period"] = trends["period"].astype(str)
    trends["total_avis"] = avis_df.groupby("period").size().values

    # Calcul des tendances (régression linéaire)
    if len(trends) > 1:
        X = np.arange(len(trends)).reshape(-1, 1)

        # Tendance satisfaction
        if "note" in avis_df.columns:
            lr_satisfaction = LinearRegression()
            lr_satisfaction.fit(X, trends["note"])
            trends["satisfaction_trend"] = lr_satisfaction.predict(X)
            satisfaction_slope = lr_satisfaction.coef_[0]
        else:
            satisfaction_slope = 0

        # Tendance volume
        lr_volume = LinearRegression()
        lr_volume.fit(X, trends["total_avis"])
        trends["volume_trend"] = lr_volume.predict(X)
        volume_slope = lr_volume.coef_[0]

        return trends, satisfaction_slope, volume_slope

    return trends, 0, 0


def create_kpi_evolution_chart(trends, metric, title):
    """Crée un graphique d'évolution des KPIs"""
    if trends is None or metric not in trends.columns:
        return None

    fig = go.Figure()

    # Ligne des données réelles
    fig.add_trace(
        go.Scatter(
            x=trends["period"],
            y=trends[metric],
            mode="lines+markers",
            name="Données réelles",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        )
    )

    # Ligne de tendance si disponible
    trend_col = f"{metric}_trend"
    if trend_col in trends.columns:
        fig.add_trace(
            go.Scatter(
                x=trends["period"],
                y=trends[trend_col],
                mode="lines",
                name="Tendance",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Période",
        yaxis_title=metric.capitalize(),
        hovermode="x unified",
        height=400,
    )

    return fig


def calculate_performance_metrics(avis_df):
    """Calcule les métriques de performance"""
    metrics = {}

    if "date" in avis_df.columns:
        # Métriques temporelles
        current_month = avis_df["date"].dt.to_period("M").max()
        previous_month = current_month - 1

        current_data = avis_df[avis_df["date"].dt.to_period("M") == current_month]
        previous_data = avis_df[avis_df["date"].dt.to_period("M") == previous_month]

        # Volume d'avis
        metrics["volume_current"] = len(current_data)
        metrics["volume_previous"] = len(previous_data)
        metrics["volume_change"] = (
            (metrics["volume_current"] - metrics["volume_previous"])
            / max(metrics["volume_previous"], 1)
        ) * 100

        # Score moyen
        if "note" in avis_df.columns:
            metrics["score_current"] = current_data["note"].mean()
            metrics["score_previous"] = previous_data["note"].mean()
            metrics["score_change"] = (
                metrics["score_current"] - metrics["score_previous"]
            )

        # Sentiments
        if "sentiment" in avis_df.columns:
            current_positive = (current_data["sentiment"] == "positif").sum()
            previous_positive = (previous_data["sentiment"] == "positif").sum()

            metrics["positive_rate_current"] = (
                current_positive / max(len(current_data), 1)
            ) * 100
            metrics["positive_rate_previous"] = (
                previous_positive / max(len(previous_data), 1)
            ) * 100
            metrics["positive_rate_change"] = (
                metrics["positive_rate_current"] - metrics["positive_rate_previous"]
            )

    return metrics


def create_satisfaction_heatmap(avis_df):
    """Crée une heatmap de satisfaction par jour et heure"""
    if "date" not in avis_df.columns or "note" not in avis_df.columns:
        return None

    # Extraire jour de la semaine et heure
    avis_df["day_of_week"] = avis_df["date"].dt.day_name()
    avis_df["hour"] = avis_df["date"].dt.hour

    # Calculer la satisfaction moyenne
    heatmap_data = avis_df.pivot_table(
        index="day_of_week", columns="hour", values="note", aggfunc="mean"
    )

    # Réorganiser les jours
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    heatmap_data = heatmap_data.reindex(day_order)

    fig = px.imshow(
        heatmap_data,
        title="Satisfaction Moyenne par Jour et Heure",
        labels=dict(x="Heure", y="Jour", color="Note moyenne"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )

    return fig


def create_trend_forecast(avis_df):
    """Crée une prévision de tendance"""
    if "date" not in avis_df.columns:
        return None

    # Données mensuelles
    monthly_data = avis_df.groupby(avis_df["date"].dt.to_period("M")).size()

    # Préparer les données pour la prévision
    X = np.arange(len(monthly_data)).reshape(-1, 1)
    y = monthly_data.values

    # Modèle de régression
    model = LinearRegression()
    model.fit(X, y)

    # Prévision pour les 6 prochains mois
    future_X = np.arange(len(monthly_data), len(monthly_data) + 6).reshape(-1, 1)
    forecast = model.predict(future_X)

    # Créer les dates futures
    last_date = monthly_data.index[-1]
    future_dates = [last_date + i for i in range(1, 7)]

    fig = go.Figure()

    # Données historiques
    fig.add_trace(
        go.Scatter(
            x=[str(d) for d in monthly_data.index],
            y=monthly_data.values,
            mode="lines+markers",
            name="Données historiques",
            line=dict(color="blue", width=3),
        )
    )

    # Prévision
    fig.add_trace(
        go.Scatter(
            x=[str(d) for d in future_dates],
            y=forecast,
            mode="lines+markers",
            name="Prévision",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Prévision du Volume d'Avis",
        xaxis_title="Mois",
        yaxis_title="Nombre d'avis",
        hovermode="x unified",
        height=500,
    )

    return fig


def main():
    st.title("📈 Tendances et KPIs Temporels")

    # Chargement des données
    with st.spinner("Chargement des données..."):
        avis_df = load_data()

    if avis_df is None:
        st.error("Impossible de charger les données")
        return

    # Sidebar avec contrôles
    st.sidebar.header("⚙️ Paramètres")

    # Sélection de la période d'analyse
    period_options = {"Mensuel": "M", "Hebdomadaire": "W", "Quotidien": "D"}

    selected_period = st.sidebar.selectbox(
        "Période d'analyse", options=list(period_options.keys()), index=0
    )

    period_code = period_options[selected_period]

    # Calcul des tendances
    trends_data = calculate_trends(avis_df, period_code)
    if (
        trends_data is not None and trends_data[0] is not None
    ):  # modified condition to check if trends_data is not None
        trends, satisfaction_slope, volume_slope = trends_data
    else:
        trends, satisfaction_slope, volume_slope = None, 0, 0

    # Métriques de performance
    performance_metrics = calculate_performance_metrics(avis_df)

    # Affichage des KPIs principaux
    st.header("🎯 KPIs de Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "volume_current" in performance_metrics:
            volume_delta = performance_metrics.get("volume_change", 0)
            st.metric(
                "Volume Avis (Mois)",
                f"{performance_metrics['volume_current']:,}",
                f"{volume_delta:+.1f}%",
            )

    with col2:
        if "score_current" in performance_metrics:
            score_delta = performance_metrics.get("score_change", 0)
            st.metric(
                "Score Moyen",
                f"{performance_metrics['score_current']:.2f}/5",
                f"{score_delta:+.2f}",
            )

    with col3:
        if "positive_rate_current" in performance_metrics:
            positive_delta = performance_metrics.get("positive_rate_change", 0)
            st.metric(
                "Taux Positif",
                f"{performance_metrics['positive_rate_current']:.1f}%",
                f"{positive_delta:+.1f}%",
            )

    with col4:
        # Indicateur de tendance globale
        if satisfaction_slope > 0.01:
            trend_status = "🟢 Amélioration"
        elif satisfaction_slope < -0.01:
            trend_status = "🔴 Dégradation"
        else:
            trend_status = "🟡 Stable"

        st.metric("Tendance Globale", trend_status, f"Pente: {satisfaction_slope:.3f}")

    # Graphiques d'évolution
    st.header("📊 Évolution Temporelle")

    if trends is not None:
        col1, col2 = st.columns(2)

        with col1:
            # Évolution du volume
            volume_chart = create_kpi_evolution_chart(
                trends, "total_avis", "Évolution du Volume d'Avis"
            )
            if volume_chart:
                st.plotly_chart(volume_chart, use_container_width=True)

        with col2:
            # Évolution de la satisfaction
            if "note" in avis_df.columns:
                satisfaction_chart = create_kpi_evolution_chart(
                    trends, "note", "Évolution de la Satisfaction"
                )
                if satisfaction_chart:
                    st.plotly_chart(satisfaction_chart, use_container_width=True)

    # Heatmap de satisfaction
    st.header("🔥 Analyse par Jour et Heure")

    heatmap_fig = create_satisfaction_heatmap(avis_df)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Prévision
    st.header("🔮 Prévision")

    forecast_fig = create_trend_forecast(avis_df)
    if forecast_fig:
        st.plotly_chart(forecast_fig, use_container_width=True)

    # Analyse comparative
    st.header("📋 Analyse Comparative")

    if "date" in avis_df.columns:
        # Comparaison par mois
        monthly_comparison = (
            avis_df.groupby(avis_df["date"].dt.to_period("M"))
            .agg(
                {
                    "note": "mean" if "note" in avis_df.columns else lambda x: 0,
                    "sentiment": lambda x: (
                        (x == "positif").sum() if "sentiment" in avis_df.columns else 0
                    ),
                }
            )
            .reset_index()
        )

        monthly_comparison["period"] = monthly_comparison["date"].astype(str)
        monthly_comparison["total_avis"] = (
            avis_df.groupby(avis_df["date"].dt.to_period("M")).size().values
        )

        if "sentiment" in avis_df.columns:
            monthly_comparison["positive_rate"] = (
                monthly_comparison["sentiment"] / monthly_comparison["total_avis"] * 100
            )

        # Tableau comparatif
        st.subheader("Comparaison Mensuelle")

        display_columns = ["period", "total_avis"]
        if "note" in avis_df.columns:
            display_columns.append("note")
        if "positive_rate" in monthly_comparison.columns:
            display_columns.append("positive_rate")

        comparison_df = monthly_comparison[display_columns].tail(12)

        # Renommer les colonnes pour l'affichage
        column_names = {
            "period": "Période",
            "total_avis": "Nombre d'avis",
            "note": "Note moyenne",
            "positive_rate": "Taux positif (%)",
        }

        comparison_df = comparison_df.rename(columns=column_names)

        st.dataframe(comparison_df, use_container_width=True)

    # Insights et recommandations
    st.header("💡 Insights et Recommandations")

    insights = []

    # Analyse des tendances
    if satisfaction_slope > 0.01:
        insights.append(
            "✅ **Tendance positive** : La satisfaction client s'améliore progressivement"
        )
    elif satisfaction_slope < -0.01:
        insights.append(
            "⚠️ **Tendance négative** : La satisfaction client se dégrade, action requise"
        )

    if volume_slope > 0:
        insights.append(
            "📈 **Volume croissant** : Le nombre d'avis augmente, signe d'engagement"
        )
    elif volume_slope < 0:
        insights.append(
            "📉 **Volume décroissant** : Baisse du nombre d'avis, vérifier l'engagement"
        )

    # Recommandations basées sur les métriques
    if "positive_rate_current" in performance_metrics:
        if performance_metrics["positive_rate_current"] < 50:
            insights.append(
                "🔴 **Action urgente** : Taux de satisfaction critique (<50%)"
            )
        elif performance_metrics["positive_rate_current"] < 70:
            insights.append(
                "🟡 **Amélioration nécessaire** : Taux de satisfaction modéré"
            )
        else:
            insights.append("🟢 **Performance solide** : Bon taux de satisfaction")

    for insight in insights:
        st.markdown(insight)

    if not insights:
        st.info("Aucun insight particulier détecté. Continuez le monitoring régulier.")


if __name__ == "__main__":
    main()
