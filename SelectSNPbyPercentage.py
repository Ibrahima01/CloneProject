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

#==============================================================================================================

# Fonction pour calculer le pourcentage de la présence du SNP chez les fumeurs
def ppsf(colonne):
    return ((main_df[colonne] != 0) & (main_df['Smoking_status'] == 1)).mean() * 100

# Fonction pour calculer le pourcentage de l'absence du SNP chez les non-fumeurs
def pasnf(colonne):
    return ((main_df[colonne] == 0) & (main_df['Smoking_status'] == 0)).mean() * 100

# Initialiser une liste pour stocker les colonnes sélectionnées
colonnes_retenues = []

# Appliquer les fonctions à toutes les colonnes
for colonne in main_df.columns:
    diff_pourcentage = abs(ppsf(colonne) - pasnf(colonne))
    if diff_pourcentage < 10:
        colonnes_retenues.append(colonne)

# Sélectionner les colonnes dont la différence de pourcentage est inférieure à 10%
df_selection = main_df[colonnes_retenues]

df_selection.to_csv("SNP_selected_by_frequence.csv", sep="\t", index=False)
