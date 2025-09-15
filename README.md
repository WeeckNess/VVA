# Victory Vision Analytics 🏎️📊

## 🎯 Objectif
Prédire les résultats des courses de Formule 1 en intégrant l’influence de la météo.  
L’idée est d’analyser si et comment les conditions météorologiques peuvent :  
- Favoriser certaines équipes ou pilotes  
- Resserrer les écarts de temps  
- Donner plus de chances aux « petites » équipes  
- Introduire un facteur d’incertitude dans les résultats  

---

## 📦 Données utilisées
- `circuits.csv`  
- `constructor_results.csv`  
- `constructor_standings.csv`  
- `daily_weather.parquet`  
- **+ autres datasets via Kaggle.com**  

---

## 🛠️ Outils
- **Pandas** → chargement, nettoyage et transformation des données  
- **Pyarrow / fastparquet** → lecture des fichiers `.parquet`  
- **Scikit-learn** → régressions, classification, modèles de prédiction  
- **Seaborn** → graphiques exploratoires  
- **Simpy / Numpy** → simulation d’événements (Grand Prix)  
- **Kaggle API** → récupération de datasets complémentaires  
- **Requests** → appels d’API météo  

---

## ✅ Livrables
1. **Dashboard interactif**  
   - Visualisation des prévisions des résultats de courses  
   - Impact de la météo intégré  

2. **Simulateur de Grand Prix**  
   - Permet d’obtenir des résultats basés sur différents scénarios  
   - Intègre des événements aléatoires influencés par la météo  