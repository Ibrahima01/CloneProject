import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2

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

#Dummy pour donner du sens au zéro
not_col= ["ID", "Smoking_status"]
df = pd.get_dummies(data= main_df, columns= [col for col in main_df.columns if col not in not_col])

#==============================================================================================================

#X = df.iloc[:, 2:]
features = df.iloc[:, 2:]
labels=df.iloc[:, 1]
# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

#==============================================================================================================

#Créer et entrqiner le modèle SelectKBest
# Create a SelectKBest feature selector and specify the number of features to select
num_features_to_select = 1000  # Specify the desired number of features
selector = SelectKBest(chi2, k=num_features_to_select)

# Fit the selector on the training data and transform the data to select the top-k features
X_train_selected = selector.fit_transform(features, labels)
#X_test_selected = selector.transform(test_features)

# Create a new DataFrame with the selected features
selected_features = features.columns[selector.get_support()]
new_data = features[selected_features]

# Now, 'new_data' contains only the selected features, and you can save it as a new dataset
#new_data['Smoking_status'] = labels  # Add the target variable back to the new dataset if needed

new_data.to_csv("1000_Best_Features.csv", sep="\t", index=False)
