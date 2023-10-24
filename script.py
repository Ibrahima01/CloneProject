import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os  # Import the os module to handle file paths
from sklearn.metrics import roc_auc_score


df_base=pd.read_csv("All_features")

df_pheno=pd.read_csv('donnees_transformees.csv', sep='\t')
df_pheno= df_pheno[["ID", "Smoking_status"]]

# Jointure des données SNP et phénotype sur la colonne "ID"
joined_data = df_base.merge(df_pheno, on="ID")

# Supprimer les lignes où la dernière colonne est égale à -1
df = joined_data[joined_data.iloc[:, -1] != -1]

scaled_features = df.iloc[:, 1:-1]

labels=df.iloc[:, -1]

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Créez un modèle SVM avec probabilité activée
model = SVC(probability=True, random_state=42)

# Créez un objet KFold pour la validation croisée à 10 plis
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Effectuez la validation croisée et obtenez les scores pour chaque pli
accuracy_scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='precision')
recall_scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='recall')
f1_scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='f1')

# Affichez les métriques moyennes et écart-types
print(f'Mean Accuracy: {accuracy_scores.mean():.2f}')
print(f'Standard Deviation (Accuracy): {accuracy_scores.std():.2f}')

print(f'Mean Precision: {precision_scores.mean():.2f}')
print(f'Standard Deviation (Precision): {precision_scores.std():.2f}')

print(f'Mean Recall: {recall_scores.mean():.2f}')
print(f'Standard Deviation (Recall): {recall_scores.std():.2f}')

print(f'Mean F1-score: {f1_scores.mean():.2f}')
print(f'Standard Deviation (F1-score): {f1_scores.std():.2f}')

# Créez un modèle SVM avec probabilité activée
model = SVC(probability=True, random_state=42)

# Créez un objet KFold pour la validation croisée à 10 plis
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialisez des listes pour stocker les valeurs fpr (taux de faux positifs) et tpr (taux de vrais positifs)
fpr_list = []
tpr_list = []
roc_auc_list = []

# Créez une figure pour afficher les courbes ROC
plt.figure(figsize=(8, 6))

for train_index, test_index in kf.split(train_features):
    X_train_cv, X_test_cv = train_features.iloc[train_index], train_features.iloc[test_index]
    y_train_cv, y_test_cv = train_labels.iloc[train_index], train_labels.iloc[test_index]

    # Entraînez le modèle SVM sur l'ensemble de formation actuel
    model.fit(X_train_cv, y_train_cv)

    # Prédire les probabilités sur l'ensemble de test actuel
    y_pred_prob = model.predict_proba(X_test_cv)[:, 1]

    # Calculez le ROC AUC
    fpr, tpr, _ = roc_curve(y_test_cv, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Ajoutez les valeurs à la liste
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

# Tracez les courbes ROC pour chaque pli
for i in range(len(fpr_list)):
    plt.plot(fpr_list[i], tpr_list[i], lw=2, label=f'Fold {i+1} (AUC = {roc_auc_list[i]:.2f})')

# Paramètres de la figure
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - 10-Fold Cross Validation')
plt.legend(loc="lower right")

# Create the directory if it doesn't exist
os.makedirs(figure_directory, exist_ok=True)

# Save the figure with a specific name (you can customize the filename)
figure_filename = os.path.join(figure_directory, 'auc_k_cross_validation.png')
plt.savefig(figure_filename)

# Close the current figure to release resources
plt.close()


# Créez une liste pour stocker le nombre de features et l'AUC correspondant
num_features = []
auc_scores = []

# Variations du nombre de features (vous pouvez ajuster la plage)
for n in range(1, train_features.shape[1] + 1):
    # Sélectionner les n premières features
    X_train_subset = train_features.iloc[:, :n]

    # Créer un objet de modèle (par exemple, SVM) à l'intérieur de la boucle
    model = SVC(probability=True, random_state=42)
    
    # Initialiser les scores pour la validation croisée
    auc_scores_fold = []

    # Créez un objet KFold pour la validation croisée
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Effectuer la validation croisée
    for train_index, test_index in kf.split(X_train_subset):
        X_train_cv, X_test_cv = X_train_subset.iloc[train_index], X_train_subset.iloc[test_index]
        y_train_cv, y_test_cv = train_labels.iloc[train_index], train_labels.iloc[test_index]

        # Entraîner le modèle sur le sous-ensemble de features
        model.fit(X_train_cv, y_train_cv)

        # Prédire les probabilités et calculer l'AUC
        y_pred_prob = model.predict_proba(X_test_cv)[:, 1]
        auc = roc_auc_score(y_test_cv, y_pred_prob)
        auc_scores_fold.append(auc)

    # Calculer la moyenne des AUC de tous les plis
    auc_mean = np.mean(auc_scores_fold)

    # Enregistrer le nombre de features et l'AUC
    num_features.append(n)
    auc_scores.append(auc_mean)

# Tracer la courbe d'évolution de l'AUC en fonction du nombre de features
plt.figure(figsize=(10, 6))
plt.plot(num_features, auc_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Nombre de Features')
plt.ylabel('AUC Score (Cross-Validated)')
plt.title('Évolution de l\'AUC en fonction du nombre de Features (Cross-Validated)')
plt.grid(True)

# Create the directory if it doesn't exist
os.makedirs(figure_directory, exist_ok=True)

# Save the figure with a specific name (you can customize the filename)
figure_filename = os.path.join(figure_directory, 'auc_vs_num_features.png')
plt.savefig(figure_filename)

# Close the current figure to release resources
plt.close()
