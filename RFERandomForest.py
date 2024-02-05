import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
# Mon DataFrame principal
main_df = pd.read_csv('https://drive.google.com/uc?id=1O4FVHjkQCwduz1a02xZFbtQh8I0v1EsP', sep='\t')

# Liste des liens vers les fichiers restants
urls = [
    'https://drive.google.com/uc?id=19r82zXqWZg_rp0xpGlNcHc4OQxItgFHX',
    'https://drive.google.com/uc?id=1NZ-qP0FX02Whjy5cb4LVlDKuBMXn8YiI',
    'https://drive.google.com/uc?id=1KVFQxtDBjLhO-s3vFk8OcyrlJqjJpmAM',
    'https://drive.google.com/uc?id=121USCGCctk8k-U5-aHlF7m0byVgrTDuE',
    'https://drive.google.com/uc?id=12jRAmwqR25i_jXsEuqFST5-Kyd2cCgbh',
    'https://drive.google.com/uc?id=1LeZ_jNFq8bPgV9aLaQY_3FftItbH5u9t',
    'https://drive.google.com/uc?id=1nOGlP9jpelIKsZDWaiovvuYvpwkL35OQ',
    'https://drive.google.com/uc?id=1sJ0_aaviYLkH4yqZ9X8i8Gm8X5_NHh12',
    'https://drive.google.com/uc?id=1tXZdO-z80-N1y7jcxARSITGznTs-fxyv',
    'https://drive.google.com/uc?id=1IodxiSiMifPoFR6AyR8BqWkz05fAueys',
    'https://drive.google.com/uc?id=1KJjjTiTsmM6K1yfjQsvZRrs7Loc-e-UG',
    'https://drive.google.com/uc?id=1AbAD2WhF2eg-lUqCv3BcXvuZ_DjJb9Vo',
    'https://drive.google.com/uc?id=1giq4_IFHyK_XerPR3WHZW-qvI-z0xDxQ',
    'https://drive.google.com/uc?id=15xMITPd4O2WNwUThxXkl6tHOqq-sA65X',
    'https://drive.google.com/uc?id=1HwascUUEPl6sHuV_Zxst8axYRhyt6nU5',
    'https://drive.google.com/uc?id=165y-VGGVQiYJCGqjsEPWF-zucw0rLrTD',
    'https://drive.google.com/uc?id=1PuIKLrjmn-b6u7weXl4TPOmTuyIJ1blL',
    'https://drive.google.com/uc?id=1s6yNL1w0nZjjNjmPdQ_T4xXFnDUZWqIT'
]

# Jointure par lot avec chaque fichier
for url in urls:
    chunk_df = pd.read_csv(url, sep='\t')  # Chargez un DataFrame à la fois
    main_df = pd.merge(main_df, chunk_df, on='ID', how='outer')  # Effectuez la jointure
    del chunk_df  # Libérez la mémoire du DataFrame chargé
    
#==============================================================================================================

# Lien des phénotypes partagé direct du fichier Google Drive
url = 'https://drive.google.com/uc?id=1B0_OYAeq5Y5cSIYCl3MGBKmxOw-fjWQu'
# Charger le fichier CSV directement depuis le lien Google Drive
pheno = pd.read_csv(url, sep='\t')
pheno= pheno[["ID", "Smoking_status"]]

# Jointure des données SNP et phénotype sur la colonne "ID"
main_df = main_df.merge(pheno, on="ID")
# Supprimer les lignes où la dernière colonne est égale à -1
main_df = main_df[main_df.iloc[:, -1] != -1]
not_col= ["ID", "Smoking_status"]
df = pd.get_dummies(data= main_df, columns= [col for col in main_df.columns if col not in not_col])


X = df.iloc[:, 2:]

y = df.iloc[:, 1]

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier as the base model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Specify the number of features to select using RFE
num_features_to_select = 10

# Initialize RFE with the model and the number of features to select
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)

# Fit RFE to the training data
rfe.fit(X_train, y_train)

# Get the ranking of each feature
feature_ranking = rfe.ranking_

# Get the selected features
selected_features = np.where(feature_ranking == 1)[0]

# Train the final model using the selected features
model.fit(X_train[:, selected_features], y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test[:, selected_features], y_test)
print("Accuracy on the Test Set:", accuracy)
