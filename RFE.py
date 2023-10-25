import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load your dataset (replace 'your_data.csv' with your actual data file)
df_01 = pd.read_csv('../01_transposed_transform.csv', sep='\t')
df_02 = pd.read_csv('../02_transposed_transform.csv', sep='\t')
df_03 = pd.read_csv('../03_transposed_transform.csv', sep='\t')
df_04 = pd.read_csv('../04_transposed_transform.csv', sep='\t')
df_05 = pd.read_csv('../05_transposed_transform.csv', sep='\t')
df_06 = pd.read_csv('../06_transposed_transform.csv', sep='\t')
df_07 = pd.read_csv('../07_transposed_transform.csv', sep='\t')
df_08 = pd.read_csv('../08_transposed_transform.csv', sep='\t')
df_09 = pd.read_csv('../09_transposed_transform.csv', sep='\t')
df_10 = pd.read_csv('../10_transposed_transform.csv', sep='\t')
df_11 = pd.read_csv('../11_transposed_transform.csv', sep='\t')
df_12 = pd.read_csv('../12_transposed_transform.csv', sep='\t')
df_13 = pd.read_csv('../13_transposed_transform.csv', sep='\t')
df_14 = pd.read_csv('../14_transposed_transform.csv', sep='\t')
df_15 = pd.read_csv('../15_transposed_transform.csv', sep='\t')
df_16 = pd.read_csv('../16_transposed_transform.csv', sep='\t')
df_17 = pd.read_csv('../17_transposed_transform.csv', sep='\t')
df_18 = pd.read_csv('../18_transposed_transform.csv', sep='\t')
df_19 = pd.read_csv('../19_transposed_transform.csv', sep='\t')
dataframes = [df_01, df_02, df_03, df_04, df_05, df_06, df_07, df_08, df_09, df_10, df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19]

df_base = df_01
# Parcourez la liste des DataFrames restants pour effectuer la jointure
for df in dataframes[1:]:
    # Utilisez la méthode merge pour effectuer une jointure sur la colonne "ID"
    df_base = df_base.merge(df, on="ID", how="inner")

df_pheno = pd.read_csv('../../Phenotype/donnees_transformees.csv', sep='\t')
df_pheno= df_pheno[["ID", "Smoking_status"]]

data = df_base.merge(df_pheno, on="ID")

# Split the data into features (X) and target (y)
X = data.iloc[:, 1:-1]  # Replace 'target_variable' with the name of your target variable
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a logistic regression model (or any other estimator you prefer)
estimator = LogisticRegression(solver='liblinear', random_state=42)

# Specify the number of features to select
num_features_to_select = 10000  # Specify the desired number of features

# Create an RFE selector with the chosen estimator and number of features to select
rfe_selector = RFE(estimator, n_features_to_select=num_features_to_select)

# Fit the RFE selector on the training data to identify the most important features
rfe_selector.fit(X_train, y_train)

# Get the selected features and their rankings
selected_features = X.columns[rfe_selector.support_]
feature_rankings = rfe_selector.ranking_

# Create a new DataFrame with the selected features
new_data = X_train[selected_features]

# Print the names of the selected features and their rankings
print("Selected Features:", selected_features)
print("Feature Rankings:", feature_rankings)

# Now, 'new_data' contains only the selected features, and you can save it as a new dataset
new_data['target_variable'] = y_train  # Add the target variable back to the new dataset if needed
new_data.to_csv('new_dataset_RFE.csv', index=False)  # Save the new dataset to a CSV file
