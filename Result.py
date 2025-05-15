import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ✅ Load features and labels
X_metrics = pd.read_csv("X_metrics.csv")
X_llm = pd.read_csv("X_llm.csv")
X_fusion = pd.read_csv("X_fusion.csv")
y = pd.read_csv("y.csv")["label"].values.ravel()
print("✅ Loaded target vector, shape:", y.shape)

# ✅ Evaluation function
def evaluate(X, y):
    print("▶️ Evaluating input shape:", X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return round(precision, 2), round(recall, 2), round(f1, 2), round(auc, 2)

# ✅ Compare core configurations
results = {
    "Metrics-only": evaluate(X_metrics, y),
    "LLM-only": evaluate(X_llm, y),
    "Fusion (Ours)": evaluate(X_fusion, y),
}

df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Precision", "Recall", "F1-score", "AUC-ROC"]
)
print("\n✅ Model performance comparison:")
print(df)

# ✅ Evaluate prompt-enhanced variants
X_llm_prompt = pd.read_csv("X_llm_prompt.csv")
X_fusion_prompt = pd.read_csv("X_fusion_prompt.csv")

results = {
    "Metrics-only": evaluate(X_metrics, y),
    "LLM-only": evaluate(X_llm, y),
    "LLM+Prompt": evaluate(X_llm_prompt, y),
    "Fusion": evaluate(X_fusion, y),
    "Fusion+Prompt": evaluate(X_fusion_prompt, y),
}

df = pd.DataFrame.from_dict(
    results, orient="index", columns=["Precision", "Recall", "F1-score", "AUC-ROC"]
)
print("\n✅ Full variant comparison with prompt optimization:")
print(df)
