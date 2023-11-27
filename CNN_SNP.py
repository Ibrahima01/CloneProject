import pandas as pd
import tensorflow as tf
import numpy as np

# Load your dataset (replace 'your_data.csv' with your actual data file)
df_01 = pd.read_csv('../../01_transposed_transform.csv', sep='\t')
df_02 = pd.read_csv('../../02_transposed_transform.csv', sep='\t')
df_03 = pd.read_csv('../../03_transposed_transform.csv', sep='\t')
df_04 = pd.read_csv('../../04_transposed_transform.csv', sep='\t')
df_05 = pd.read_csv('../../05_transposed_transform.csv', sep='\t')
df_06 = pd.read_csv('../../06_transposed_transform.csv', sep='\t')
df_07 = pd.read_csv('../../07_transposed_transform.csv', sep='\t')
df_08 = pd.read_csv('../../08_transposed_transform.csv', sep='\t')
df_09 = pd.read_csv('../../09_transposed_transform.csv', sep='\t')
df_10 = pd.read_csv('../../10_transposed_transform.csv', sep='\t')
df_11 = pd.read_csv('../../11_transposed_transform.csv', sep='\t')
df_12 = pd.read_csv('../../12_transposed_transform.csv', sep='\t')
df_13 = pd.read_csv('../../13_transposed_transform.csv', sep='\t')
df_14 = pd.read_csv('../../14_transposed_transform.csv', sep='\t')
df_15 = pd.read_csv('../../15_transposed_transform.csv', sep='\t')
df_16 = pd.read_csv('../../16_transposed_transform.csv', sep='\t')
df_17 = pd.read_csv('../../17_transposed_transform.csv', sep='\t')
df_18 = pd.read_csv('../../18_transposed_transform.csv', sep='\t')
df_19 = pd.read_csv('../../19_transposed_transform.csv', sep='\t')
dataframes = [df_01, df_02, df_03, df_04, df_05, df_06, df_07, df_08, df_09, df_10, df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19]

df_base = df_01
# Parcourez la liste des DataFrames restants pour effectuer la jointure
for df in dataframes[1:]:
    # Utilisez la méthode merge pour effectuer une jointure sur la colonne "ID"
    df_base = df_base.merge(df, on="ID", how="inner")

df_pheno = pd.read_csv('../../../Phenotype/donnees_transformees.csv', sep='\t')
df_pheno= df_pheno[["ID", "Smoking_status"]]

data = df_base.merge(df_pheno, on="ID")

SNP_data = data.iloc[:, 1:-1]  # Exemple de données SNP
phenotypes = data.iloc[:, -1]   # Exemple de labels phénotype

# Diviser les données en ensembles d'entraînement et de test
train_size = int(0.8 * len(SNP_data))
test_size = len(SNP_data) - train_size

train_images = SNP_data[:train_size]
train_labels = phenotypes[:train_size]

test_images = SNP_data[train_size:]
test_labels = phenotypes[train_size:]

# Normaliser les données
train_images = train_images.astype('float32') / 2.0  # Comme les valeurs de SNP sont 0, 1, 2, on divise par 2
test_images = test_images.astype('float32') / 2.0

nombre_de_colonnes = df.shape[1]
nombre_de_SNP= nombre_de_colonnes -2

# Créer un modèle CNN adapté à vos données SNP
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((nombre_de_SNP, 1), input_shape=(nombre_de_SNP,)),  # Adapter la taille en fonction du nombre de SNP
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
# Entraîner le modèle
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Évaluer le modèle sur les données de test
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy on test set: {test_acc}')

# Utiliser tf.GradientTape() pour calculer les gradients par rapport aux entrées (SNPs)
with tf.GradientTape() as tape:
    inputs = tf.convert_to_tensor(test_images)  # Convertir les données de test en tenseur TensorFlow
    tape.watch(inputs)  # Surveiller les entrées pour enregistrer les gradients
    predictions = model(inputs)  # Prédictions du modèle sur les données de test

# Obtenir les gradients par rapport aux entrées (SNPs)
gradients = tape.gradient(predictions, inputs).numpy()

# Calculer l'importance des SNPs en utilisant la magnitude des gradients
snp_importance = np.mean(np.abs(gradients), axis=0)  # Importance moyenne pour chaque SNP

# Trier les SNPs en fonction de leur importance
num_top_snps = 1000  # Nombre de SNPs les plus influents à afficher
top_snps_indices = np.argsort(snp_importance)[::-1][:num_top_snps]  # Indices des SNPs les plus influents

# Afficher les indices des SNPs les plus influents
print("Indices des SNPs les plus influents :", top_snps_indices)

# Enregistrer les SNPs les plus influents en local
top_snps = SNP_data.columns[top_snps_indices]  # Récupérer les données des SNPs les plus influents
np.savetxt('top_1000_snps.csv', top_snps, delimiter=',')  # Enregistrer les SNPs dans un fichier CSV



