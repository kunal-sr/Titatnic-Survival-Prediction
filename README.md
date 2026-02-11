## 🚢 Titanic - Machine Learning from Disaster

This repository contains my solution for the Titanic - Machine Learning from Disaster competition on Kaggle.

The goal of this competition is to predict which passengers survived the Titanic shipwreck using machine learning techniques.

---

# 🔗 Competition Link:
**[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)**

---

```

📂 Repository Structure
├── train.csv              # Training dataset
├── test.csv               # Test dataset
├── gender_submission.csv  # Kaggle baseline submission
├── main.py                # Model training & prediction script
├── submission.csv         # My final Kaggle submission
└── README.md
```

---

# 🧠 Approach

1️⃣ Data Preprocessing

- Combined train and test datasets for consistent preprocessing
- Extracted Title from passenger names
- Grouped rare titles into a single category (Rare)
- Standardized similar titles (e.g., Mlle → Miss, Mme → Mrs)

2️⃣ Missing Value Handling

- Age filled using median age grouped by passenger Title (smarter imputation)
- Fare filled using median
- Embarked filled using mode

3️⃣ Feature Engineering

- Created FamilySize feature:
- FamilySize = SibSp + Parch + 1

---

# Created IsAlone feature:

```
IsAlone = 1 if FamilySize == 1 else 0
```

--- 

# One-hot encoding for:

- Title
- Embarked

---

# Converted:

Sex → numerical (male=0, female=1)

---

# Dropped unused columns:

- Name
- Ticket
- Cabin

  ---

# 🤖 Model Used

- Random Forest Classifier


```
RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42
)
```

- Why Random Forest?
- Handles non-linearity well
- Robust to overfitting (with depth control)
- Performs strongly on structured/tabular data

  ---

# 🏁 Submission

Predictions were generated on the processed test set and saved to:

```
submission.csv
```

Format:

```
PassengerId,Survived
892,0
893,1
...
```

---

# 📊 Results

Model: Random Forest

- Engineered features significantly improved prediction quality
- Title-based age imputation improved model stability
- (You can add your Kaggle score here if you'd like.)

  ---

# 🚀 How to Run

Clone the repository:

```
git clone https://github.com/yourusername/titanic-ml.git
```

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib
```

Run the script:

```
python main.py
```
---

The submission file will be generated automatically.

# 📌 Key Learnings

- Feature engineering matters more than model complexity.
- Smart missing value handling improves performance.
- Random Forest is a powerful baseline for tabular ML problems.

Kaggle competitions are great for practical ML experience.
