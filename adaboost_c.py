import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Our variable target outcome: Outcome = 1 Diabetes, Outcome = 0 No diabetes
# drop the label for de model 
X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=25,
)


# Scaling (fit only on train)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# AdaBoost Classifier
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.7,
    random_state=25
)

# Train model
adaboost.fit(X_train_scaled, y_train)

# Predictions
y_preds = adaboost.predict(X_test_scaled)

# Evaluation
print("AdaBoost Classifier Results:\n")
print(classification_report(y_test, y_preds))
