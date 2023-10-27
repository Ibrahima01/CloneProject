import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os
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
    
df_pheno = pd.read_csv('../../Phenotype/donnees_transformees.csv', sep='\t')
df_pheno= df_pheno[["ID", "Smoking_status"]]

data_ = df_base.merge(df_pheno, on="ID")

# Supprimer les lignes où la dernière colonne est égale à -1
data_ = data_[data_.iloc[:, -1] != -1]
not_col= ["ID", "Smoking_status"]
df = pd.get_dummies(data=data_, columns= [col for col in data_.columns if col not in not_col])

X = df.iloc[:, 2:]

scaled_features = X

labels=df.iloc[:, 1]

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

model_LR=LogisticRegression(max_iter=10000)
model_LR.fit(train_features, train_labels)

# Prédire les labels pour l'ensemble de test
predictions = model_LR.predict(test_features)
# Calculer l'accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy of Logistic Regression:", accuracy)

# Calculer la précision
precision_LR = precision_score(test_labels, predictions, average='macro')
print("Precision Logistic regression:", precision_LR)

# Calculer le rappel
recall_LR = recall_score(test_labels, predictions, average='macro')
print("Recall Logistic regression:", recall_LR)

# Calculer le F1-score
f1_LR = f1_score(test_labels, predictions, average='macro')
print("F1-Score Logistic regression:", f1_LR)

# Get feature importances from the Logistic Regression model
importance_scores = model_LR.coef_[0]

# Sort the features based on their importance scores
sorted_indices = np.argsort(importance_scores)

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = importance_scores[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)

top_100_features_df.to_csv("TopLR.txt", index=False, sep ="\t")
=========================================================================================================================
# Créer et entraîner le modèle SVM
model_SVM_linear = svm.SVC(kernel='linear', probability=True, random_state=42)
model_SVM_linear.fit(train_features, train_labels)
print ("SVM fiting is OK")

# Prédire les labels pour l'ensemble de test
predictions_SVM = model_SVM_linear.predict(test_features)
# Calculer l'accuracy
accuracy_SVM = accuracy_score(test_labels, predictions_SVM)
print("Accuracy SVM:", accuracy_SVM)

# Get feature importances from the linear SVM model (approximated by coefficients)
importance_scores = model_SVM_linear.coef_.ravel()  # For linear kernel, importance is approximated by coefficients

# Sort the features based on their importance scores
sorted_indices = np.argsort(importance_scores)

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = importance_scores[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)
top_100_features_df.to_csv("TopSVM.txt", index=False, sep ="\t")
=========================================================================================================================



# Create and train the Decision Tree model
model_decision_tree = DecisionTreeClassifier(random_state=42)
model_decision_tree.fit(train_features, train_labels)

# Predict labels for the test set
predictions_decision_tree = model_decision_tree.predict(test_features)

# Calculate accuracy
accuracy_decision_tree = accuracy_score(test_labels, predictions_decision_tree)
print("Accuracy Decision Tree:", accuracy_decision_tree)

# Calculate precision
precision_decision_tree = precision_score(test_labels, predictions_decision_tree, average='macro')
print("Precision Decision Tree:", precision_decision_tree)

# Get feature importances from the Decision Tree model
importance_scores = model_decision_tree.feature_importances_

# Sort the features based on their importance scores
sorted_indices = importance_scores.argsort()[::-1]

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = importance_scores[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)
top_100_features_df.to_csv("TopDT.txt", index=False, sep ="\t")
=========================================================================================================================

# Create and train the Random Forest model
model_random_forest = RandomForestClassifier(random_state=42)
model_random_forest.fit(train_features, train_labels)

# Predict labels for the test set
predictions_random_forest = model_random_forest.predict(test_features)

# Calculate accuracy
accuracy_random_forest = accuracy_score(test_labels, predictions_random_forest)
print("Accuracy Random Forest:", accuracy_random_forest)

# Calculate precision
precision_random_forest = precision_score(test_labels, predictions_random_forest, average='macro')
print("Precision Random Forest:", precision_random_forest)

# Get feature importances from the Random Forest model
importance_scores = model_random_forest.feature_importances_

# Sort the features based on their importance scores
sorted_indices = importance_scores.argsort()[::-1]

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = importance_scores[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)
top_100_features_df.to_csv("TopRF.txt", index=False, sep ="\t")
=========================================================================================================================

# Create individual classifiers
model_svm = SVC(kernel='linear', probability=True, random_state=42)
model_lr = LogisticRegression(max_iter=10000)
model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)

# Create a Voting Classifier with the individual models
voting_classifier = VotingClassifier(estimators=[
    ('svm', model_svm),
    ('lr', model_lr),
    ('dt', model_dt),
    ('rf', model_rf)
], voting='soft')

# Train the Voting Classifier
voting_classifier.fit(train_features, train_labels)

# Predict labels for the test set
predictions_voting = voting_classifier.predict(test_features)

# Calculate accuracy
accuracy_voting = accuracy_score(test_labels, predictions_voting)
print("Accuracy Voting Classifier:", accuracy_voting)

# Calculate precision
precision_voting = precision_score(test_labels, predictions_voting, average='macro')
print("Precision Voting Classifier:", precision_voting)

# Train each of the individual models on the full dataset
model_svm.fit(scaled_features, labels)
model_lr.fit(scaled_features, labels)
model_dt.fit(scaled_features, labels)
model_rf.fit(scaled_features, labels)

# Collect the feature importances from each individual model
importance_svm = model_svm.coef_.ravel() if hasattr(model_svm, "coef_") else None
importance_lr = model_lr.coef_[0] if hasattr(model_lr, "coef_") else None
importance_dt = model_dt.feature_importances_ if hasattr(model_dt, "feature_importances_") else None
importance_rf = model_rf.feature_importances_ if hasattr(model_rf, "feature_importances_") else None

# Aggregate the importance scores
importance_scores = []
if importance_svm is not None:
    importance_scores.append(importance_svm)
if importance_lr is not None:
    importance_scores.append(importance_lr)
if importance_dt is not None:
    importance_scores.append(importance_dt)
if importance_rf is not None:
    importance_scores.append(importance_rf)

# Calculate the mean importance score across models
mean_importance = np.mean(importance_scores, axis=0)

# Sort the features based on their importance scores
sorted_indices = np.argsort(mean_importance)[::-1]

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = mean_importance[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)
top_100_features_df.to_csv("TopVoting.txt", index=False, sep ="\t")
=========================================================================================================================

# Create and train the XGBoost model
model_xgboost = xgb.XGBClassifier()
model_xgboost.fit(train_features, train_labels)

# Predict labels for the test set
predictions_xgboost = model_xgboost.predict(test_features)

# Calculate accuracy
accuracy_xgboost = accuracy_score(test_labels, predictions_xgboost)
print("Accuracy XGBoost:", accuracy_xgboost)

# Calculate precision
precision_xgboost = precision_score(test_labels, predictions_xgboost, average='macro')
print("Precision XGBoost:", precision_xgboost)

# Get feature importances from the XGBoost model
importance_scores = model_xgboost.feature_importances_

# Sort the features based on their importance scores
sorted_indices = importance_scores.argsort()[::-1]

# Display the top 100 features with their importance scores
top_100_features = X.columns[sorted_indices[:100]]
top_100_importance_scores = importance_scores[sorted_indices[:100]]

# Create a DataFrame for easy display
top_100_features_df = pd.DataFrame({'Feature': top_100_features, 'Importance Score': top_100_importance_scores})

# Display the top 100 features with their importance scores
print(top_100_features_df)
top_100_features_df.to_csv("TopXGBoost.txt", index=False, sep ="\t")
=========================================================================================================================


# Create instances of the models
model_LR = LogisticRegression(max_iter=10000)
model_SVM = SVC()
model_DT = DecisionTreeClassifier(random_state=42)
model_RF = RandomForestClassifier(random_state=42)
model_Voting = VotingClassifier(estimators=[
    ('lr', model_LR),
    ('dt', model_DT),
    ('rf', model_RF)
], voting='soft')
model_XGBoost = xgb.XGBClassifier()

models = [model_LR, model_SVM, model_DT, model_RF, model_Voting, model_XGBoost]
model_names = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'Voting Classifier', 'XGBoost']

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for model, name in zip(models, model_names):
    scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='accuracy')
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    print(f'{name} - Mean Accuracy: {mean_accuracy:.2f}')
    print(f'{name} - Standard Deviation: {std_accuracy:.2f}')

