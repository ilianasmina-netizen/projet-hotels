import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Configuration générale de la page web
st.set_page_config(
    page_title="Pilotage Hôtelier & Prévision des Annulations",
    layout="wide"
)

# Fonctions de mise en forme des nombres à la française (espaces et virgules)
def fr(val, dec=0):
    if dec == 0:
        return f"{int(round(val)):,}".replace(",", " ")
    return f"{val:,.{dec}f}".replace(",", " ").replace(".", ",")

def pc(val):
    return f"{val * 100:.1f} %".replace(".", ",")


# ==============================================================================
# PARTIE 1 : CHARGEMENT ET NETTOYAGE DES DONNÉES (ÉTAPE 1 DU NOTEBOOK)
# ==============================================================================
@st.cache_data
def charger_donnees():
    # Lecture du fichier CSV placé dans le même dossier que app.py
    df = pd.read_csv("reservations_hotels.csv")
    
    # Correction des valeurs manquantes dans 'enfants' (remplacées par 0)
    df["enfants"] = df["enfants"].fillna(0)
    
    # Suppression des réservations impossibles (0 occupant ou prix négatif)
    cond_sans_occupant = (df["adultes"] + df["enfants"]) == 0
    cond_prix_negatif = df["prix_nuit"] < 0
    propres = df[~(cond_sans_occupant | cond_prix_negatif)].copy()
    
    # Calcul du chiffre d'affaires potentiel de chaque réservation et du mois d'arrivée
    propres["ca"] = propres["prix_nuit"] * propres["nuits"]
    propres["annee_mois"] = propres["date_arrivee"].str.slice(0, 7)
    
    return propres

propres = charger_donnees()


# ==============================================================================
# PARTIE 2 : ENTRAÎNEMENT DU MODÈLE DE PRÉVISION (ÉTAPE 3 DU NOTEBOOK)
# ==============================================================================
@st.cache_resource
def entrainer_modele(df):
    # Sélection stricte des 13 variables connues dès la réservation (aucune fuite de données)
    variables = [
        "hotel", "mois", "delai_reservation", "nuits", "adultes", "enfants",
        "segment", "type_client", "client_habitue", "annulations_precedentes",
        "depot", "prix_nuit", "demandes_speciales"
    ]
    colonnes_quali = ["hotel", "mois", "segment", "type_client", "depot"]
    
    # Transformation des variables qualitatives en indicatrices (0 ou 1)
    X = pd.get_dummies(df[variables], columns=colonnes_quali, dtype=int)
    y = df["annulee"]
    
    # Découpage chronologique : apprentissage sur les arrivées avant le 1er janvier 2017
    avant_2017 = df["date_arrivee"] < "2017-01-01"
    X_train = X[avant_2017]
    y_train = y[avant_2017]
    
    # Entraînement de la régression logistique
    modele = LogisticRegression(max_iter=3000)
    modele.fit(X_train, y_train)
    
    return modele, X.columns.tolist(), variables, colonnes_quali

modele, colonnes_X, variables_modele, colonnes_quali = entrainer_modele(propres)


# ==============================================================================
# PARTIE 3 : BARRE LATÉRALE DE FILTRES
# ==============================================================================
st.sidebar.header("Filtres du tableau de bord")

# Filtre 1 : Établissement (Lisbonne / Algarve)
hotels_dispo = ["Tous les hôtels"] + sorted(propres["hotel"].unique().tolist())
choix_hotel = st.sidebar.selectbox("Hôtel", hotels_dispo)

# Filtre 2 : Année d'arrivée
annees_dispo = ["Toutes les années"] + sorted(propres["annee"].astype(str).unique().tolist())
choix_annee = st.sidebar.selectbox("Année d'arrivée", annees_dispo)

# Filtre 3 : Segment de clientèle
segments_dispo = sorted(propres["segment"].unique().tolist())
choix_segments = st.sidebar.multiselect(
    "Segments de clientèle",
    options=segments_dispo,
    default=segments_dispo
)

# Application des filtres choisis sur les données propres
df_filtre = propres.copy()
if choix_hotel != "Tous les hôtels":
    df_filtre = df_filtre[df_filtre["hotel"] == choix_hotel]
if choix_annee != "Toutes les années":
    df_filtre = df_filtre[df_filtre["annee"].astype(str) == choix_annee]
if choix_segments:
    df_filtre = df_filtre[df_filtre["segment"].isin(choix_segments)]


# ==============================================================================
# PARTIE 4 : AFFICHAGE DU TABLEAU DE BORD ET DE L'OUTIL DE PRÉVISION
# ==============================================================================
st.title("Tableau de bord : Pilotage hôtelier & Prévision des annulations")
st.markdown(
    "Suivi de l'activité commerciale des hôtels de **Lisbonne** et d'**Algarve** "
    "et évaluation du risque d'annulation dès la réservation."
)

onglet_kpi, onglet_prevision = st.tabs([
    "Indicateurs de gestion (Étape 2)",
    "Outil de prévision des annulations (Étape 3)"
])

# ------------------------------------------------------------------------------
# ONGLET 1 : INDICATEURS DE GESTION ET GRAPHIQUES
# ------------------------------------------------------------------------------
with onglet_kpi:
    if len(df_filtre) == 0:
        st.warning("Aucune réservation ne correspond aux filtres sélectionnés.")
    else:
        realisees = df_filtre[df_filtre["annulee"] == 0]
        annulees = df_filtre[df_filtre["annulee"] == 1]

        # Calcul des 5 indicateurs de base
        taux_annul = df_filtre["annulee"].mean()
        ca_realise = realisees["ca"].sum()
        nuits_vendues = realisees["nuits"].sum()
        prix_moyen = ca_realise / nuits_vendues if nuits_vendues > 0 else 0
        ca_annule = annulees["ca"].sum()

        # Calcul des 4 indicateurs complémentaires
        part_famille = (realisees["enfants"] > 0).mean() if len(realisees) > 0 else 0
        part_fidele = (realisees["client_habitue"] == 1).mean() if len(realisees) > 0 else 0
        alos = realisees["nuits"].mean() if len(realisees) > 0 else 0
        ca_non_couvert = annulees[annulees["depot"] != "Non remboursable"]["ca"].sum()
        taux_perte_seche = (ca_non_couvert / ca_annule) if ca_annule > 0 else 0

        # Affichage ligne 1 : les 5 indicateurs principaux
        st.subheader("Indicateurs clés d'activité et de chiffre d'affaires")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Taux d'annulation", pc(taux_annul))
        c2.metric("CA réalisé", f"{fr(ca_realise)} €")
        c3.metric("Nuits vendues", fr(nuits_vendues))
        c4.metric("Prix moyen / nuit", f"{fr(prix_moyen, 2)} €")
        c5.metric("CA annulé", f"{fr(ca_annule)} €")

        # Affichage ligne 2 : les 4 indicateurs complémentaires
        st.subheader("Profil de la clientèle et exposition financière")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clientèle familiale", pc(part_famille))
        k2.metric("Clientèle fidèle", pc(part_fidele))
        k3.metric("Durée moyenne (ALOS)", f"{fr(alos, 2)} nuits")
        k4.metric("Taux de perte sèche", pc(taux_perte_seche))

        st.divider()

        # Graphiques ligne 1 : CA mensuel et taux d'annulation par délai
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**Évolution mensuelle du CA réalisé (€)**")
            ca_mensuel = realisees.groupby("annee_mois")["ca"].sum()
            fig1, ax1 = plt.subplots(figsize=(7, 3.8))
            ca_mensuel.plot(kind="line", marker="o", color="#1f4e78", ax=ax1)
            ax1.set_xlabel("Mois d'arrivée")
            ax1.set_ylabel("CA réalisé (€)")
            ax1.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_g2:
            st.markdown("**Taux d'annulation selon le délai de réservation (%)**")
            bornes = [-1, 7, 30, 90, 180, 800]
            etiquettes = ["0-7 j", "8-30 j", "31-90 j", "91-180 j", "+180 j"]
            df_filtre["classe_delai"] = pd.cut(
                df_filtre["delai_reservation"], bins=bornes, labels=etiquettes
            )
            taux_delai = df_filtre.groupby("classe_delai", observed=False)["annulee"].mean() * 100
            fig2, ax2 = plt.subplots(figsize=(7, 3.8))
            taux_delai.plot(kind="bar", color="#2e75b6", ax=ax2)
            ax2.set_xlabel("Délai avant l'arrivée")
            ax2.set_ylabel("Taux d'annulation (%)")
            ax2.grid(axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig2)

        # Graphiques ligne 2 : taux d'annulation par dépôt et par segment
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            st.markdown("**Taux d'annulation selon le type de dépôt (%)**")
            taux_depot = df_filtre.groupby("depot")["annulee"].mean().sort_values() * 100
            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            taux_depot.plot(kind="barh", color="#5b9bd5", ax=ax3)
            ax3.set_xlabel("Taux d'annulation (%)")
            ax3.set_ylabel("")
            ax3.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_g4:
            st.markdown("**Taux d'annulation selon le segment (%)**")
            taux_seg = df_filtre.groupby("segment")["annulee"].mean().sort_values() * 100
            fig4, ax4 = plt.subplots(figsize=(7, 3.5))
            taux_seg.plot(kind="barh", color="#ed7d31", ax=ax4)
            ax4.set_xlabel("Taux d'annulation (%)")
            ax4.set_ylabel("")
            ax4.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig4)

# ------------------------------------------------------------------------------
# ONGLET 2 : OUTIL DE PRÉVISION D'UNE RÉSERVATION
# ------------------------------------------------------------------------------
with onglet_prevision:
    st.subheader("Évaluer le risque d'annulation d'une nouvelle réservation")
    st.markdown(
        "Le modèle utilise uniquement les informations connues **au moment de la réservation**. "
        "Le seuil par défaut est fixé à **0,60 (60 %)** afin de limiter les fausses alertes de surréservation."
    )

    seuil = st.slider(
        "Seuil d'alerte choisi",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05
    )

    with st.form("form_prevision"):
        col1, col2, col3 = st.columns(3)

        with col1:
            in_hotel = st.selectbox("Hôtel", sorted(propres["hotel"].unique()))
            in_mois = st.selectbox("Mois d'arrivée", sorted(propres["mois"].unique()))
            in_delai = st.number_input("Délai de réservation (jours)", min_value=0, max_value=750, value=60)
            in_nuits = st.number_input("Nombre de nuits", min_value=1, max_value=60, value=3)

        with col2:
            in_segment = st.selectbox("Segment", sorted(propres["segment"].unique()))
            in_type_client = st.selectbox("Type de client", sorted(propres["type_client"].unique()))
            in_depot = st.selectbox("Type de dépôt", sorted(propres["depot"].unique()))
            in_prix = st.number_input("Prix moyen par nuit (€)", min_value=0.0, max_value=5000.0, value=100.0, step=5.0)

        with col3:
            in_adultes = st.number_input("Nombre d'adultes", min_value=1, max_value=10, value=2)
            in_enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0)
            in_habitue = st.selectbox("Client habitué ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            in_annul_prec = st.number_input("Annulations précédentes", min_value=0, max_value=30, value=0)
            in_demandes = st.number_input("Demandes spéciales", min_value=0, max_value=5, value=0)

        bouton = st.form_submit_button("Calculer la probabilité d'annulation")

    if bouton:
        nouvelle_resa = pd.DataFrame([{
            "hotel": in_hotel,
            "mois": in_mois,
            "delai_reservation": in_delai,
            "nuits": in_nuits,
            "adultes": in_adultes,
            "enfants": in_enfants,
            "segment": in_segment,
            "type_client": in_type_client,
            "client_habitue": in_habitue,
            "annulations_precedentes": in_annul_prec,
            "depot": in_depot,
            "prix_nuit": in_prix,
            "demandes_speciales": in_demandes
        }])

        # Même encodage que lors de l'entraînement du modèle
        X_new = pd.get_dummies(nouvelle_resa[variables_modele], columns=colonnes_quali, dtype=int)
        X_new = X_new.reindex(columns=colonnes_X, fill_value=0)

        proba_annul = modele.predict_proba(X_new)[0, 1]

        st.divider()
        r1, r2 = st.columns([1, 2])
        r1.metric("Probabilité d'annulation", pc(proba_annul))

        if proba_annul >= seuil:
            r2.error(
                f"**Alerte annulation (probabilité ≥ {pc(seuil)}) :** risque élevé détecté. "
                "Recommandation : relance client préventive pour confirmer la venue ou demande de garantie."
            )
        else:
            r2.success(
                f"**Réservation fiable (probabilité < {pc(seuil)}) :** risque maîtrisé. "
                "Recommandation : maintien normal sans surréservation."
            )
