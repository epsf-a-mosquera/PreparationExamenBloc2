
Analyse & limites — Modèle de régression (order_basket_value)

Les métriques obtenues sont exceptionnellement élevées (R² ≈ 0.9995, MAE ≈ 2€). Ce niveau de performance suggère fortement la présence d’une variable très directement corrélée ou dérivée de la cible, typiquement une variable de type total de commande (ex. order_order_total) pouvant être une combinaison de composantes proches de order_basket_value (panier, remises, frais). Dans un contexte “production”, il est indispensable de vérifier la définition métier de ces champs et d’exclure toute feature dépendante de la cible afin d’éviter le target leakage.
Enfin, la présence de fortes colinéarités (ex. entre order_order_total, order_discount, order_shipping_fee) peut rendre l’interprétation des coefficients instable : le modèle reste performant mais les coefficients doivent être analysés avec prudence.

Analyse & limites — Modèle de classification (order_is_returned)

Le jeu de données est déséquilibré (≈ 22.6% de retours). Le modèle de régression logistique obtient une accuracy proche de la classe majoritaire, mais au seuil de décision 0.5 il prédit uniquement la classe 0 (aucun retour détecté), ce qui entraîne precision/recall/F1 nuls pour la classe 1. Cependant, les scores AUC (ROC AUC ≈ 0.61, PR AUC ≈ 0.34) indiquent que le modèle possède une capacité de discrimination : il classe partiellement les exemples, mais le seuil choisi n’est pas adapté. En pratique, on améliore ce comportement en ajustant le seuil (selon l’objectif métier : maximiser recall ou F1) et/ou en utilisant class_weight="balanced" afin de pénaliser davantage les erreurs sur la classe minoritaire.

(Option bonus, 2 lignes) Recommandations rapides

Régression : tester une version sans order_order_total (et/ou variables directement dérivées) pour valider l’absence de fuite de cible.

Classification : tester class_weight="balanced" et sélectionner un seuil optimisé (F1/recall) plutôt que 0.5.