# Guide de Déploiement sur Render 🚀

Ce document explique comment déployer l'application de Dashboard Actuariel SAAQ sur **Render** (une plateforme Cloud très adaptée pour Streamlit).

## Prérequis

1. Le code a été poussé sur votre dépôt Github (le dossier `Données` contenant les CSV doit impérativement être inclus dans le dépôt car le code a besoin de ces données pour fonctionner).
2. Un compte sur [Render.com](https://render.com/).

## Étapes de Déploiement

### 1. Créer un nouveau Web Service sur Render

1. Connectez-vous à votre tableau de bord Render.
2. Cliquez sur le bouton **"New +"** en haut à droite.
3. Sélectionnez **"Web Service"**.
4. Connectez votre compte GitHub si ce n'est pas déjà fait, puis choisissez le dépôt Git contenant ce code.

### 2. Configuration du Web Service

Remplissez les informations de configuration de la manière suivante :

- **Name** : `dashboard-actuariel-saaq` (ou le nom de votre choix)
- **Region** : Choisissez la région la plus proche de vos utilisateurs (ex: `Ohio (US East)` ou `Frankfurt (EU Central)`).
- **Branch** : `main` (ou la branche contenant votre code).
- **Root Directory** : Laissez vide (ou mettez le chemin vers le dossier si l'application n'est pas à la racine).
- **Runtime** : `Python 3`
- **Build Command** :
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command** :
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```

### 3. Choix de l'Instance (Pricing)

- Vous pouvez choisir le plan **Free** pour des tests.
- _Note_ : Sur le plan gratuit, l'application se mettra en veille après 15 minutes d'inactivité et mettra environ 50 secondes à redémarrer lors de la prochaine visite. Les algorithmes K-Means réels de la page 4 demandant un peu de mémoire, un plan "Starter" ou "Standard" (RAM >= 512MB) offrira de meilleures performances.

### 4. Variables d'Environnement (Optionnel)

Aucune variable d'environnement complexe n'est requise pour cette application telle quelle. Toutefois, si vous utilisez une version spécifique de Python, vous pouvez forcer la version de Render en ajoutant la variable :

- `PYTHON_VERSION` : `3.9.6` (ou la version correspondant à votre environnement local).

### 5. Finaliser

1. Cliquez sur le bouton **"Create Web Service"** en bas de la page.
2. Render va maintenant récupérer votre code depuis GitHub, installer les librairies du fichier `requirements.txt` et lancer Streamlit.
3. Les logs s'afficheront dans la console Render. Une fois que vous voyez `Live`, votre application est en ligne !

Vous pourrez accéder au tableau de bord via l'URL générée par Render (ex: `https://dashboard-actuariel-saaq.onrender.com`).
