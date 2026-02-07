# Tarea -----------------------------------------------------------------
# Usar modelo adaboost classifier
# cargar data set
# usar scaler
# run model - .fit, .predict
# ver su comportamiento con classification_report
# -----------------------------------------------------------------------


# This dataset is originally  from the National Institute of Diabetes and Digestive and Kidney Diseases. 
# The objective of the dataset is to diagnostically predict whether a patient has 
# diabetes, based on certain diagnostic measurements included in the dataset.

# dataset in data/diabetes.csv

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_wine

wine_data = load_wine()

# convert data to pd dataframe
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# add the target level
# wine_df["target"] = wine_data.target

# preprocessing
# missing values, redundant values, outliers, errors, noise

X = wine_df[wine_data.feature_names].copy()
y = wine_df["proline"].copy()

scaler = StandardScaler()
scaler.fit(X)

X_scaled = scaler.transform(X.values)

print(X_scaled[0])

X_train_sacaled, X_test_sacaled, y_train, t_test = train_test_split(
    X_scaled, y, train_size=0.7, random_state=25
)

print(
    f"Train size: {round(len(X_train_sacaled)/len(X)*100)}%\nTest size: {round(len(X_test_sacaled)/len(X)*100)}%"
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

logistic_regression.fit(X_train_sacaled, y_train)
svm.fit(X_train_sacaled, y_train)
tree.fit(X_train_sacaled, y_train)

log_reg_preds = logistic_regression.predict(X_test_sacaled)
svm_preds = svm.predict(X_test_sacaled)
tree_preds = tree.predict(X_test_sacaled)

from sklearn.metrics import classification_report

model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds,
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
