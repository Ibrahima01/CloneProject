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
    
df_snp=pd.read_csv("699_SNPs_KEGG__sans_occ.txt")

jointure = pd.merge(main_df, df_snp, on="ID", how="right")

jointure.to_csv("699_SNPs_KEGG_inverse_dummy", sep="\t", index=False)
