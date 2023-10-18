import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# Charger les données à partir du fichier CSV
df_01=pd.read_csv('Transformed data/01_transposed_transform.csv', sep='\t')
df_02=pd.read_csv('Transformed data/02_transposed_transform.csv', sep='\t')
df_03=pd.read_csv('Transformed data/03_transposed_transform.csv', sep='\t')
df_04=pd.read_csv('Transformed data/04_transposed_transform.csv', sep='\t')
df_05=pd.read_csv('Transformed data/05_transposed_transform.csv', sep='\t')
df_06=pd.read_csv('Transformed data/06_transposed_transform.csv', sep='\t')
df_07=pd.read_csv('Transformed data/07_transposed_transform.csv', sep='\t')
df_08=pd.read_csv('Transformed data/08_transposed_transform.csv', sep='\t')
df_09=pd.read_csv('Transformed data/09_transposed_transform.csv', sep='\t')
df_10=pd.read_csv('Transformed data/10_transposed_transform.csv', sep='\t')
df_11=pd.read_csv('Transformed data/11_transposed_transform.csv', sep='\t')
df_12=pd.read_csv('Transformed data/12_transposed_transform.csv', sep='\t')
df_13=pd.read_csv('Transformed data/13_transposed_transform.csv', sep='\t')
df_14=pd.read_csv('Transformed data/14_transposed_transform.csv', sep='\t')
df_15=pd.read_csv('Transformed data/15_transposed_transform.csv', sep='\t')
df_16=pd.read_csv('Transformed data/16_transposed_transform.csv', sep='\t')
df_17=pd.read_csv('Transformed data/17_transposed_transform.csv', sep='\t')
df_18=pd.read_csv('Transformed data/18_transposed_transform.csv', sep='\t')
df_19=pd.read_csv('Transformed data/19_transposed_transform.csv', sep='\t')

# Jointure des données partitions SNPs sur la colonne "ID"
joined_data1 = df_01.merge(df_02, on="ID")
joined_data2 = joined_data1.merge(df_03, on="ID")
joined_data3 = joined_data2.merge(df_04, on="ID")
joined_data4 = joined_data3.merge(df_05, on="ID")
joined_data5 = joined_data4.merge(df_06, on="ID")
joined_data6 = joined_data5.merge(df_07, on="ID")
joined_data7 = joined_data6.merge(df_08, on="ID")
joined_data8 = joined_data7.merge(df_09, on="ID")
joined_data9 = joined_data8.merge(df_10, on="ID")
joined_data10 = joined_data9.merge(df_11, on="ID")
joined_data11 = joined_data10.merge(df_12, on="ID")
joined_data12 = joined_data11.merge(df_13, on="ID")
joined_data13 = joined_data12.merge(df_14, on="ID")
joined_data14 = joined_data13.merge(df_15, on="ID")
joined_data15 = joined_data14.merge(df_16, on="ID")
joined_data16 = joined_data15.merge(df_17, on="ID")
joined_data17 = joined_data16.merge(df_18, on="ID")
joined_data18 = joined_data17.merge(df_19, on="ID")
pheno_data = pd.read_csv("Phenotype/donnees_transformees.csv", sep="\t")
pheno_data = pheno_data[["ID", "Smoking_status"]]
joined_data = joined_data18.merge(pheno_data, on="ID")


scaler = StandardScaler()
scaled_features = scaler.fit_transform(joined_data.iloc[:, 1:-1])

labels=joined_data.iloc[:, -1]

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Créer et entraîner le modèle SVM
model_SVM_linear = svm.SVC(kernel='linear')
model_SVM_linear.fit(train_features, train_labels)
print ("Done")

# Prédire les labels pour l'ensemble de test
predictions_SVM = model_SVM_linear.predict(test_features)
# Calculer l'accuracy
accuracy_SVM = accuracy_score(test_labels, predictions_SVM)
print("Accuracy SVM:", accuracy_SVM)

# Obtenir les coefficients du modèle SVM
coefficients = model_SVM_linear.coef_

# Calculer la somme des carrés des coefficients pour normaliser
coef_sum_squares = np.sum(coefficients**2, axis=0)
norm_coef_sum_squares = coef_sum_squares / np.sum(coef_sum_squares)

# Trier les indices des caractéristiques par ordre décroissant de l'importance
top_feature_indices = np.argsort(norm_coef_sum_squares)[::-1][:10]

# Obtenir les noms des caractéristiques (variants)
feature_names = joined_data.columns[1:-1]

# Afficher les 10 meilleurs variants avec leurs pourcentages d'importance
print("Top 10 Variants and their Importance of SVM:")
for idx in top_feature_indices:
    print(f"{feature_names[idx]} : {norm_coef_sum_squares[idx]*100:.2f}%")


from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

model_LR=LogisticRegression(max_iter=10000)
model_LR.fit(train_features, train_labels)

# Prédire les labels pour l'ensemble de test
predictions = model_LR.predict(test_features)

# Calculer l'accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy of Logistic Regression:", accuracy)

# Obtenir les coefficients du modèle de régression logistique
coefficients = model_LR.coef_

# Calculer la somme des carrés des coefficients pour normaliser
coef_sum_squares = np.sum(coefficients**2, axis=0)
norm_coef_sum_squares = coef_sum_squares / np.sum(coef_sum_squares)

# Trier les indices des caractéristiques par ordre décroissant de l'importance
top_feature_indices = np.argsort(norm_coef_sum_squares)[::-1][:10]

# Obtenir les noms des caractéristiques (variants)
feature_names = joined_data.columns[1:-1]

# Afficher les 10 meilleurs variants avec leurs pourcentages d'importance
print("Top 10 Variants and their Importance (Logistic Regression):")
for idx in top_feature_indices:
    print(f"{feature_names[idx]} : {norm_coef_sum_squares[idx]*100:.2f}%")
    

from sklearn.tree import DecisionTreeClassifier

# Créer et entraîner le modèle DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train_features, train_labels)

# Faire des prédictions sur les données de test
predictions = model.predict(test_features)

# Calculer l'exactitude des prédictions
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy of decision Tree Classifier: {:.2f}%".format(accuracy * 100))

# Obtenir l'importance des caractéristiques du modèle Decision Tree
feature_importances = model.feature_importances_

# Trier les indices des caractéristiques par ordre décroissant de l'importance
top_feature_indices = np.argsort(feature_importances)[::-1][:10]

# Obtenir les noms des caractéristiques (variants)
feature_names = joined_data.columns[1:-1]

# Afficher les 10 meilleurs variants avec leurs pourcentages d'importance
print("Top 10 Variants and their Importance (Decision Tree Classifier):")
for idx in top_feature_indices:
    print(f"{feature_names[idx]} : {feature_importances[idx]*100:.2f}%")


from sklearn.ensemble import RandomForestClassifier

# Créer et entraîner le modèle RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_features, train_labels)

#Faire des prédictions sur l'ensemble de test
predictions = rf_classifier.predict(test_features)
#Évaluer les performances du modèle
accuracy = (predictions == test_labels).mean()
print("Accuracy of Random Forest:", accuracy)

# Obtenir l'importance des caractéristiques du modèle Random Forest
feature_importances = rf_classifier.feature_importances_

# Trier les indices des caractéristiques par ordre décroissant de l'importance
top_feature_indices = np.argsort(feature_importances)[::-1][:10]

# Obtenir les noms des caractéristiques (variants)
feature_names = joined_data.columns[1:-1]

# Afficher les 10 meilleurs variants avec leurs pourcentages d'importance
print("Top 10 Variants and their Importance (Random Forest Classifier):")
for idx in top_feature_indices:
    print(f"{feature_names[idx]} : {feature_importances[idx]*100:.2f}%")
    

from sklearn.ensemble import VotingClassifier


svm_model = svm.SVC()
logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()
mlp_model = MLPClassifier()

#Regrouper les modèles dans un ensemble learning
ensemble_model = VotingClassifier(
estimators=[
("svm", svm_model),
("logistic", logistic_model),
("decision_tree", decision_tree_model),
("random_forest", random_forest_model),
("mlp", mlp_model),
]
)

#Entraîner l'ensemble learning
ensemble_model.fit(train_features, train_labels)

#Faire des prédictions sur l'ensemble de test
predictions = ensemble_model.predict(test_features)
#Évaluer les performances de l'ensemble learning
accuracy = (predictions == test_labels).mean()
print("Accuracy of Voting :", accuracy)

# Initialiser un dictionnaire pour stocker les importances des caractéristiques
feature_importances = {}

# Obtenir les importances des caractéristiques pour chaque modèle individuel
for name, model_voting in ensemble_model.named_estimators_.items():
    if hasattr(model_voting, "feature_importances_"):
        feature_importances[name] = model_voting.feature_importances_

# Calculer les importances agrégées en moyenne
aggregate_importances = np.mean(list(feature_importances.values()), axis=0)

# Trier les indices des caractéristiques par ordre décroissant des importances
top_feature_indices = np.argsort(aggregate_importances)[::-1][:10]

# Obtenir les noms des caractéristiques (variants)
feature_names = joined_data.columns[1:-1]

# Afficher les 10 meilleurs variants avec leurs importances agrégées
print("Top 10 Variants and their Aggregate Importances (Voting Classifier):")
for idx in top_feature_indices:
    print(f"{feature_names[idx]} : {aggregate_importances[idx]:.4f}")
    

from sklearn.ensemble import AdaBoostClassifier

# Initialisation du AdaBoostClassifier
boosting_model = AdaBoostClassifier(
    base_estimator=None,  # Vous pouvez spécifier un modèle de base si nécessaire
    n_estimators=50,  # Nombre d'estimateurs (modèles) à entraîner
    random_state=42  # Réglez la graine aléatoire pour la reproductibilité
)

# Entraîner le modèle AdaBoost avec les modèles individuels
boosting_model.fit(train_features, train_labels)

# Faire des prédictions sur l'ensemble de test
predictions = boosting_model.predict(test_features)

# Évaluer les performances du modèle
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy of Boosting :", accuracy)

# Obtenir l'importance des caractéristiques (features)
feature_importance = boosting_model.feature_importances_
# Obtenir les noms des colonnes (variantes)
variant_names = joined_data.columns[1:-1]  # Exclure la première colonne (ID) et la dernière colonne (labels)

# Trier les variantes par importance décroissante
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_variants = variant_names[sorted_indices]

# Afficher les 10 variantes les plus importantes avec leurs taux d'importance
top_10_variants = sorted_variants[:10]
top_10_importances = feature_importance[sorted_indices][:10]

print("Les 10 variantes les plus importantes avec leurs taux d'importance:")
for variant, importance in zip(top_10_variants, top_10_importances):
    print(f"{variant}: {importance}")
    


