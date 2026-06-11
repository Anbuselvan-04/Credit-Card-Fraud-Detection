# ==========================================================
# CREDIT CARD FRAUD DETECTION USING VOTING CLASSIFIER
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from imblearn.over_sampling import SMOTE

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("\nLoading Dataset...")

data = pd.read_csv("Creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# ==========================================================
# 2. TRAIN TEST SPLIT
# ==========================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 3. SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================================
# 4. SMOTE
# ==========================================================

print("Balancing Dataset using SMOTE...")

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

# ==========================================================
# 5. MODELS
# ==========================================================

print("Training Models...")

lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================================
# 6. VOTING CLASSIFIER
# ==========================================================

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('rf', rf)
    ],
    voting='soft'
)

voting_clf.fit(
    X_train_resampled,
    y_train_resampled
)

# Individual Models

lr.fit(X_train_resampled, y_train_resampled)
dt.fit(X_train_resampled, y_train_resampled)
rf.fit(X_train_resampled, y_train_resampled)

# ==========================================================
# 7. TEST PREDICTIONS
# ==========================================================

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

voting_pred = voting_clf.predict(X_test)

# ==========================================================
# 8. ACCURACY CALCULATION
# ==========================================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Voting Classifier"
]

accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, voting_pred)
]

# ==========================================================
# 9. REPORT FUNCTION
# ==========================================================

def predict_transaction(raw_sample):

    sample = scaler.transform([raw_sample])

    lr_p = lr.predict(sample)[0]
    dt_p = dt.predict(sample)[0]
    rf_p = rf.predict(sample)[0]

    final_p = voting_clf.predict(sample)[0]

    prob = voting_clf.predict_proba(sample)[0]

    print("\n")
    print("=" * 54)
    print("        CREDIT CARD ANALYSIS REPORT")
    print("=" * 54)

    if final_p == 1:
        print("\nTransaction Status: ❌ FRAUD DETECTED")
    else:
        print("\nTransaction Status: ✅ LEGITIMATE")

    print("\nModel Confidence:")

    print(
        "- Logistic Regression:",
        "Fraud" if lr_p == 1 else "Normal"
    )

    print(
        "- Decision Tree:",
        "Fraud" if dt_p == 1 else "Normal"
    )

    print(
        "- Random Forest:",
        "Fraud" if rf_p == 1 else "Normal"
    )

    print(
        "\nFinal Decision (Voting Classifier):",
        "FRAUD ❌" if final_p == 1 else "LEGITIMATE ✅"
    )

    print("\nProbability Score:")

    print(
        "- Fraud:",
        round(prob[1], 2)
    )

    print(
        "- Normal:",
        round(prob[0], 2)
    )

    print("\nRecommendation:")

    if final_p == 1:

        print("⚠️ Block transaction immediately")
        print("⚠️ Send alert to customer")
        print("⚠️ Flag account for review")

    else:

        print("✅ Allow transaction")
        print("✅ No suspicious activity detected")

    print("=" * 54)

# ==========================================================
# 10. USER CHOICE
# ==========================================================

print("\nChoose Transaction Type")
print("1. Random Legitimate Transaction")
print("2. Random Fraud Transaction")

choice = input("\nEnter Choice (1 or 2): ")

if choice == "1":

    legit_rows = y_test[y_test == 0].index

    idx = np.random.choice(legit_rows)

    sample = X_test_raw.loc[idx].values

    predict_transaction(sample)

elif choice == "2":

    fraud_rows = y_test[y_test == 1].index

    idx = np.random.choice(fraud_rows)

    sample = X_test_raw.loc[idx].values

    predict_transaction(sample)

else:

    print("Invalid Choice")
    exit()

# ==========================================================
# 11. WAIT BEFORE SHOWING GRAPHS
# ==========================================================

input(
    "\nPress ENTER to view performance graphs..."
)

# ==========================================================
# 12. ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

bars = plt.bar(
    models,
    accuracies
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy Score"
)

for i, val in enumerate(accuracies):

    plt.text(
        i,
        val + 0.001,
        f"{val*100:.2f}%",
        ha='center'
    )

plt.ylim(0.95,1.0)

plt.show()

# ==========================================================
# 13. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    voting_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Legitimate','Fraud'],
    yticklabels=['Legitimate','Fraud']
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# ==========================================================
# 14. ROC CURVE
# ==========================================================

y_prob = voting_clf.predict_proba(
    X_test
)[:,1]

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.show()

# ==========================================================
# 15. CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        voting_pred
    )
)

print("\nProject Execution Completed Successfully.")