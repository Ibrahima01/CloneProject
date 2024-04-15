import pandas as pd


# Mon DataFrame principal
main_df = pd.read_csv('../Transformed data/01_transposed_transform.csv', sep='\t')

# Liste des liens vers les fichiers restants
urls = [
    '../Transformed data/02_transposed_transform.csv',
    '../Transformed data/03_transposed_transform.csv',
    '../Transformed data/04_transposed_transform.csv',
    '../Transformed data/05_transposed_transform.csv',
    '../Transformed data/06_transposed_transform.csv',
    '../Transformed data/07_transposed_transform.csv',
    '../Transformed data/08_transposed_transform.csv',
    '../Transformed data/09_transposed_transform.csv',
    '../Transformed data/10_transposed_transform.csv',
    '../Transformed data/11_transposed_transform.csv',
    '../Transformed data/12_transposed_transform.csv',
    '../Transformed data/13_transposed_transform.csv',
    '../Transformed data/14_transposed_transform.csv',
    '../Transformed data/15_transposed_transform.csv',
    '../Transformed data/16_transposed_transform.csv',
    '../Transformed data/17_transposed_transform.csv',
    '../Transformed data/18_transposed_transform.csv',
    '../Transformed data/19_transposed_transform.csv'
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
    presence=ppsf(colonne)
    absence=pasnf(colonne)
    if (((presence>40) or (absence>40)) and (abs(presence-absence) > 10)):
        colonnes_retenues.append(colonne)

# Sélectionner les colonnes dont la différence de pourcentage est inférieure à 10%
df_selection = main_df[colonnes_retenues]

df_selection.to_csv("SNP_selected_by_frequence.csv", sep="\t", index=False)
