import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gdown
from sklearn.model_selection import train_test_split
import numpy as np



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

#Définition des features et des labels
features = main_df.iloc[:, 1:-1]
labels=main_df.iloc[:, -1]

#==============================================================================================================

# Récupérer le nombre de colonnes du DataFrame
num_columns = features.shape[1]

# Trouver les diviseurs du nombre de colonnes et choisir m et n
factors = []
for i in range(1, int(np.sqrt(num_columns)) + 1):
    if num_columns % i == 0:
        factors.append((i, num_columns // i))

# Choisir m et n qui minimisent la différence entre eux
m, n = factors[len(factors) // 2]  # Choix du facteur le plus proche

# Redimensionner chaque ligne en une image de dimensions m x n
num_samples = len(features)
images = features.to_numpy().reshape(num_samples, m, n, -1)

# Vérifier les dimensions des images obtenues
print(f"Dimensions des images : {images.shape}")

#==============================================================================================================

# Diviser les données en ensembles d'entraînement et de test
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normaliser les données
train_images = train_images.astype('float32') / 2.0  # Comme les valeurs de SNP sont 0, 1, 2, on divise par 2
test_images = test_images.astype('float32') / 2.0

# Créer un modèle CNN simple avec TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(m, n, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes pour non-fumeurs et fumeurs
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Évaluer le modèle
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy on test set: {test_acc}')
