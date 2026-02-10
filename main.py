import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
print(test.shape)

test_ids = test["PassengerId"]

# Combine train and test
full = pd.concat([train, test], sort=False)

# --------------------
# 🔹 Extract Title
# --------------------
full["Title"] = full["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Group rare titles
rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
               "Major", "Rev", "Sir", "Jonkheer", "Dona"]

full["Title"] = full["Title"].replace(rare_titles, "Rare")

full["Title"] = full["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})

# --------------------
# 🔥 Smarter Age Filling (NEW PART)
# --------------------
full["Age"] = full.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# --------------------
# Handle other missing values
# --------------------
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# One-hot encode Title (AFTER age filling)
full = pd.get_dummies(full, columns=["Title"], drop_first=True)

# Convert categorical variables
full["Sex"] = full["Sex"].map({"male": 0, "female": 1})
full = pd.get_dummies(full, columns=["Embarked"], drop_first=True)

# Feature Engineering
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# Drop unused columns
full.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Split back
train_processed = full[full["Survived"].notnull()]
test_processed = full[full["Survived"].isnull()].drop("Survived", axis=1)

X = train_processed.drop("Survived", axis=1)
y = train_processed["Survived"]

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42
)

model.fit(X, y)

# Predict
predictions = model.predict(test_processed)

# Create submission
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions.astype(int)
})

submission.to_csv("submission.csv", index=False)

print(submission.shape)
print(submission.head())