import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

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

# Fit logistic regression model with p-values
model = sm.Logit(y_train, sm.add_constant(X_train))  # Add a constant for the intercept
results = model.fit()

# Get p-values for each feature
p_values = results.pvalues

# Define a significance level (e.g., 0.01 or 0.05)
significance_level = 0.01

# Select features with p-values less than the significance level
selected_features = X_train.columns[p_values < significance_level]

# Create a new DataFrame with the selected features
new_data = X_train[selected_features]

# Now, 'new_data' contains only the selected features, and you can save it as a new dataset
new_data['target_variable'] = y_train  # Add the target variable back to the new dataset if needed
new_data.to_csv('new_dataset_LR1.csv', index=False)  # Save the new dataset to a CSV file




# LASSO (L1 regularization) regression with lambda value of 10^(-3)
lasso_model = LogisticRegression(penalty='l1', C=0.001, solver='liblinear', random_state=42)
lasso_model.fit(X_train, y_train)

# Create a feature selector based on the LASSO model
lasso_selector = SelectFromModel(lasso_model, prefit=True)

# Transform the training and testing data to keep only the selected features
X_train_selected = lasso_selector.transform(X_train)
X_test_selected = lasso_selector.transform(X_test)

# Create a new DataFrame with the selected features
selected_features = X.columns[lasso_selector.get_support()]
new_data = pd.DataFrame(data=X_train_selected, columns=selected_features)

# Print the names of the selected features
print("Selected Features:", selected_features)

# Now, 'new_data' contains only the selected features, and you can save it as a new dataset
new_data['target_variable'] = y_train  # Add the target variable back to the new dataset if needed
new_data.to_csv('new_dataset_LASSO3.csv', index=False)  # Save the new dataset to a CSV file
