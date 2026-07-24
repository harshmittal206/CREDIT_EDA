"""
OPTIONAL helper script.

This is NOT run automatically — it's a template showing how to produce
model.pkl and vectorizer.pkl (here, a ColumnTransformer used as a
preprocessor/"vectorizer") from your cleaned application_data.

Edit the feature lists and model choice to match what you actually want,
then run:

    python train_model.py

It will write model.pkl and vectorizer.pkl into this folder, ready to
drop next to app.py for the Streamlit app.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------------------------------------------------
# 1. Load your cleaned data (reuse the cleaning steps from Credit_EDA__1_.ipynb)
# -----------------------------------------------------------------------
df = pd.read_csv("application_data.csv")

# TODO: apply the same cleaning/imputation/binning steps used in the EDA
# notebook here so training data matches what the app will send at
# inference time.

target_col = "TARGET"

numeric_features = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "YEARS_BIRTH", "YEARS_EMPLOYED", "CNT_FAM_MEMBERS",
]
categorical_features = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE", "CODE_GENDER",
]

X = df[numeric_features + categorical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------
# 2. Preprocessor ("vectorizer.pkl")
# -----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ]
)

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

# -----------------------------------------------------------------------
# 3. Model ("model.pkl")
# -----------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train_t, y_train)

preds = model.predict(X_test_t)
proba = model.predict_proba(X_test_t)[:, 1]
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, proba))

# -----------------------------------------------------------------------
# 4. Save artifacts
# -----------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Saved model.pkl and vectorizer.pkl")
