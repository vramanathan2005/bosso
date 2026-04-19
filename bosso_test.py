# ============================================================
# COLAB SCRIPT: 3-MODEL INTERVIEW NLP PROJECT
# ONE FILE ONLY:
#   BOSSO Interview Master Log.xlsx
#
# MODELS:
# 1. Question clustering (TF-IDF + KMeans)
# 2. Answer length regression (Ridge)
# 3. Player archetype clustering (PCA + KMeans)
# ============================================================

# =========================
# 1. IMPORTS
# =========================
import os
import re
import json
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import plotly.express as px
import plotly.graph_objects as go

from google.colab import files

pd.set_option("display.max_colwidth", 300)
np.random.seed(42)

# =========================
# 2. LOAD EXCEL FILE
# =========================
excel_filename = "BOSSO Interview Master Log.xlsx"

if not os.path.exists(excel_filename):
    raise FileNotFoundError(
        f"Could not find '{excel_filename}' in the current Colab working directory. "
        "Upload it in the Colab Files pane first."
    )

xls = pd.ExcelFile(excel_filename)
print("Loaded file:", excel_filename)
print("Sheets found:", xls.sheet_names)

dfs = []
for sheet in xls.sheet_names:
    df_sheet = pd.read_excel(excel_filename, sheet_name=sheet)
    df_sheet["sheet_name"] = str(sheet)
    dfs.append(df_sheet)

df = pd.concat(dfs, ignore_index=True)

# =========================
# 3. BASIC CLEANING
# =========================
expected_cols = [
    "season",
    "player_name",
    "interview_date",
    "source_link",
    "cleaned_question_text",
    "cleaned_player_answer_text",
    "word_count"
]

for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan

df["season"] = pd.to_numeric(df["season"], errors="coerce")
df["player_name"] = df["player_name"].astype(str).str.strip()
df["cleaned_question_text"] = df["cleaned_question_text"].fillna("").astype(str).str.strip()
df["cleaned_player_answer_text"] = df["cleaned_player_answer_text"].fillna("").astype(str).str.strip()
df["source_link"] = df["source_link"].fillna("").astype(str).str.strip()
df["interview_date"] = pd.to_datetime(df["interview_date"], errors="coerce")
df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")

def safe_word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text).lower()))

df["computed_word_count"] = df["cleaned_player_answer_text"].apply(safe_word_count)
df["word_count_final"] = np.where(
    df["word_count"].fillna(0) > 0,
    df["word_count"].fillna(0),
    df["computed_word_count"]
)

# keep usable rows
df = df[
    (df["player_name"] != "") &
    (df["player_name"].str.lower() != "nan") &
    (df["cleaned_question_text"].str.len() > 5) &
    (df["cleaned_player_answer_text"].str.len() > 10)
].copy()

print("\nUsable rows after cleaning:", len(df))
display(df[[
    "season", "player_name", "cleaned_question_text",
    "cleaned_player_answer_text", "word_count_final"
]].head(10))

# =========================
# 4. MODEL 1 — QUESTION CLUSTERING
# =========================
# Cluster unique questions to create automatic question categories

question_df = (
    df[["cleaned_question_text"]]
    .drop_duplicates()
    .reset_index(drop=True)
    .copy()
)

# Remove extremely generic / tiny questions if any slipped through
question_df = question_df[question_df["cleaned_question_text"].str.len() > 5].reset_index(drop=True)

question_vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=1,
    max_df=0.9
)
X_q = question_vectorizer.fit_transform(question_df["cleaned_question_text"])

candidate_k = [3, 4, 5, 6]
q_scores = {}
for k in candidate_k:
    if len(question_df) > k:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_q)
        q_scores[k] = silhouette_score(X_q, labels)

print("\nQuestion-cluster silhouette scores by k:")
print(q_scores)

best_q_k = max(q_scores, key=q_scores.get)
print("Chosen number of question clusters:", best_q_k)

question_kmeans = KMeans(n_clusters=best_q_k, random_state=42, n_init=20)
question_df["question_cluster"] = question_kmeans.fit_predict(X_q)

# Inspect top terms per cluster
q_feature_names = np.array(question_vectorizer.get_feature_names_out())
question_cluster_terms = {}

for c in sorted(question_df["question_cluster"].unique()):
    idx = question_df["question_cluster"] == c
    mean_tfidf = np.asarray(X_q[idx.values].mean(axis=0)).ravel()
    top_terms = q_feature_names[np.argsort(mean_tfidf)[-8:]][::-1]
    question_cluster_terms[c] = top_terms.tolist()

# Heuristic names for question clusters
def name_question_cluster(terms):
    joined = " ".join(terms)

    if any(x in joined for x in ["mistake", "turnover", "error", "adversity", "tough", "respond"]):
        return "mistake_adversity"
    if any(x in joined for x in ["leader", "leadership", "team", "guys", "locker", "culture"]):
        return "leadership_team"
    if any(x in joined for x in ["prepare", "preparation", "practice", "week", "process"]):
        return "preparation_process"
    if any(x in joined for x in ["opponent", "defense", "offense", "game", "matchup", "quarterback"]):
        return "opponent_game_context"
    return "performance"

question_cluster_name_map = {
    c: name_question_cluster(terms)
    for c, terms in question_cluster_terms.items()
}

question_df["pred_question_type"] = question_df["question_cluster"].map(question_cluster_name_map)

df = df.merge(question_df, on="cleaned_question_text", how="left")

print("\nMODEL 1: QUESTION CLUSTERING COMPLETE")
print("Question type counts:")
print(df["pred_question_type"].value_counts(dropna=False))

question_cluster_summary = pd.DataFrame({
    "question_cluster": list(question_cluster_terms.keys()),
    "question_type_name": [question_cluster_name_map[c] for c in question_cluster_terms.keys()],
    "top_terms": [": ".join(question_cluster_terms[c][:6]) for c in question_cluster_terms.keys()]
}).sort_values("question_cluster")

display(question_cluster_summary)

# =========================
# 5. ANSWER-LEVEL FEATURE ENGINEERING
# =========================
SELF_WORDS = {"i", "me", "my", "mine", "myself"}
TEAM_WORDS = {"we", "us", "our", "ours", "ourselves", "team"}
HEDGE_WORDS = {
    "maybe", "kind", "sort", "probably", "possibly", "guess",
    "perhaps", "think", "feel", "felt", "might", "little"
}
CONFIDENCE_WORDS = {
    "definitely", "obviously", "confident", "ready", "always",
    "never", "absolutely", "really", "sure"
}
GRATITUDE_WORDS = {"thank", "blessed", "grateful", "honor", "honored", "appreciate"}
ACCOUNTABILITY_PHRASES = [
    "starts with me",
    "my fault",
    "on me",
    "need to be better",
    "i need to",
    "i have to",
    "got to play better"
]
COACHSPEAK_PHRASES = [
    "control what we can control",
    "one day at a time",
    "next man up",
    "play fast",
    "stay focused",
    "execute",
    "get better"
]

def tokenize_text(text: str):
    return re.findall(r"[a-z']+", str(text).lower())

def phrase_count(text: str, phrases):
    low = str(text).lower()
    return sum(low.count(p) for p in phrases)

def extract_text_features(text: str):
    toks = tokenize_text(text)
    n = len(toks)
    if n == 0:
        return {
            "self_rate": 0.0,
            "team_rate": 0.0,
            "hedge_rate": 0.0,
            "confidence_rate": 0.0,
            "gratitude_rate": 0.0,
            "accountability_rate": 0.0,
            "coachspeak_rate": 0.0,
            "type_token_ratio": 0.0
        }

    c = Counter(toks)
    return {
        "self_rate": sum(c[w] for w in SELF_WORDS) / n,
        "team_rate": sum(c[w] for w in TEAM_WORDS) / n,
        "hedge_rate": sum(c[w] for w in HEDGE_WORDS) / n,
        "confidence_rate": sum(c[w] for w in CONFIDENCE_WORDS) / n,
        "gratitude_rate": sum(c[w] for w in GRATITUDE_WORDS) / n,
        "accountability_rate": phrase_count(text, ACCOUNTABILITY_PHRASES) / n,
        "coachspeak_rate": phrase_count(text, COACHSPEAK_PHRASES) / n,
        "type_token_ratio": len(set(toks)) / n
    }

feature_df = df["cleaned_player_answer_text"].apply(extract_text_features).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# =========================
# 6. MODEL 2 — ANSWER LENGTH REGRESSION
# =========================
X_reg = df[[
    "cleaned_question_text",
    "player_name",
    "season",
    "pred_question_type"
]].copy()

y_reg = df["word_count_final"].astype(float)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

text_feature = "cleaned_question_text"
categorical_features = ["player_name", "pred_question_type"]
numeric_features = ["season"]

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words="english"
            ),
            text_feature
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        )
    ]
)

ridge_model = Pipeline(steps=[
    ("preprocessor", reg_preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_model.fit(X_train_reg, y_train_reg)
reg_preds = ridge_model.predict(X_test_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
mae = mean_absolute_error(y_test_reg, reg_preds)

print("\nMODEL 2: RIDGE REGRESSION FOR ANSWER LENGTH")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))

reg_results = pd.DataFrame({
    "actual_word_count": y_test_reg.values,
    "pred_word_count": reg_preds
})
display(reg_results.head(20))

# add predictions for all rows
df["pred_word_count"] = ridge_model.predict(X_reg)

# =========================
# 7. MODEL 3 — PLAYER ARCHETYPE CLUSTERING
# =========================
def aggregate_player(group: pd.DataFrame):
    all_text = " ".join(group["cleaned_player_answer_text"].astype(str).tolist())

    mode_q = group["pred_question_type"].mode()
    dominant_question_type = mode_q.iloc[0] if not mode_q.empty else "unknown"

    return pd.Series({
        "n_responses": len(group),
        "n_seasons": group["season"].nunique(),
        "avg_word_count": group["word_count_final"].mean(),
        "sd_word_count": group["word_count_final"].std(ddof=0) if len(group) > 1 else 0.0,
        "avg_pred_word_count": group["pred_word_count"].mean(),
        "self_rate": group["self_rate"].mean(),
        "team_rate": group["team_rate"].mean(),
        "hedge_rate": group["hedge_rate"].mean(),
        "confidence_rate": group["confidence_rate"].mean(),
        "gratitude_rate": group["gratitude_rate"].mean(),
        "accountability_rate": group["accountability_rate"].mean(),
        "coachspeak_rate": group["coachspeak_rate"].mean(),
        "type_token_ratio": group["type_token_ratio"].mean(),
        "dominant_question_type": dominant_question_type,
        "all_text": all_text
    })

player_df = df.groupby("player_name").apply(aggregate_player).reset_index()

style_cols = [
    "n_responses",
    "n_seasons",
    "avg_word_count",
    "sd_word_count",
    "avg_pred_word_count",
    "self_rate",
    "team_rate",
    "hedge_rate",
    "confidence_rate",
    "gratitude_rate",
    "accountability_rate",
    "coachspeak_rate",
    "type_token_ratio"
]

player_text_vectorizer = TfidfVectorizer(
    max_features=150,
    ngram_range=(1, 2),
    stop_words="english"
)
X_text_cluster = player_text_vectorizer.fit_transform(player_df["all_text"]).toarray()

scaler = StandardScaler()
X_style_cluster = scaler.fit_transform(player_df[style_cols])

X_cluster = np.hstack([X_style_cluster, X_text_cluster])

candidate_k_player = [3, 4, 5, 6]
player_scores = {}
for k in candidate_k_player:
    if len(player_df) > k:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_cluster)
        player_scores[k] = silhouette_score(X_cluster, labels)

print("\nPlayer-cluster silhouette scores by k:")
print(player_scores)

best_player_k = max(player_scores, key=player_scores.get)
print("Chosen number of player clusters:", best_player_k)

player_kmeans = KMeans(n_clusters=best_player_k, random_state=42, n_init=20)
player_df["cluster"] = player_kmeans.fit_predict(X_cluster)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cluster)
player_df["pc1"] = coords[:, 0]
player_df["pc2"] = coords[:, 1]

cluster_summary = player_df.groupby("cluster")[style_cols].mean()

def label_player_cluster(row, cluster_summary_df):
    labels = []

    if row["avg_word_count"] >= cluster_summary_df["avg_word_count"].quantile(0.75):
        labels.append("Detailed")
    if row["avg_word_count"] <= cluster_summary_df["avg_word_count"].quantile(0.25):
        labels.append("Concise")
    if row["team_rate"] >= cluster_summary_df["team_rate"].quantile(0.75):
        labels.append("Team-oriented")
    if row["accountability_rate"] >= cluster_summary_df["accountability_rate"].quantile(0.75):
        labels.append("Accountability-first")
    if row["confidence_rate"] >= cluster_summary_df["confidence_rate"].quantile(0.75):
        labels.append("Confident")
    if row["coachspeak_rate"] >= cluster_summary_df["coachspeak_rate"].quantile(0.75):
        labels.append("Coach-speak")

    if len(labels) == 0:
        labels.append("Balanced")

    return " / ".join(labels[:2])

cluster_labels = {
    cluster_id: label_player_cluster(row, cluster_summary)
    for cluster_id, row in cluster_summary.iterrows()
}
player_df["archetype"] = player_df["cluster"].map(cluster_labels)

print("\nMODEL 3: PLAYER ARCHETYPE CLUSTERING")
display(
    player_df[[
        "player_name",
        "cluster",
        "archetype",
        "n_responses",
        "avg_word_count",
        "dominant_question_type"
    ]].sort_values(["cluster", "n_responses"], ascending=[True, False])
)

# =========================
# 8. VISUALS
# =========================
fig_q = px.bar(
    question_cluster_summary,
    x="question_cluster",
    y=[1] * len(question_cluster_summary),
    color="question_type_name",
    hover_data=["top_terms"],
    title="Model 1: Automatic Question Clusters"
)
fig_q.update_layout(yaxis_visible=False, yaxis_showticklabels=False)
fig_q.show()

fig_word = px.box(
    df,
    x="pred_question_type",
    y="word_count_final",
    title="Model 2: Answer Length by Predicted Question Type"
)
fig_word.show()

fig_player = px.scatter(
    player_df,
    x="pc1",
    y="pc2",
    color="archetype",
    size="n_responses",
    hover_name="player_name",
    hover_data=["avg_word_count", "n_seasons", "dominant_question_type"],
    title="Model 3: Player Communication Archetypes"
)
fig_player.show()

# =========================
# 9. SAVE OUTPUTS
# =========================
rows_output = "interview_rows_with_auto_question_types.csv"
players_output = "player_archetype_profiles.csv"
question_output = "question_cluster_summary.csv"
reg_output = "regression_test_results.csv"

df.to_csv(rows_output, index=False)
player_df.to_csv(players_output, index=False)
question_cluster_summary.to_csv(question_output, index=False)
reg_results.to_csv(reg_output, index=False)

print("\nSaved files:")
print("-", rows_output)
print("-", players_output)
print("-", question_output)
print("-", reg_output)

files.download(rows_output)
files.download(players_output)
files.download(question_output)
files.download(reg_output)

print("""
DONE.

You now have:
1. Model 1: Question clustering (automatic question categories)
2. Model 2: Ridge regression for answer length
3. Model 3: Player archetype clustering

Output files:
- interview_rows_with_auto_question_types.csv
- player_archetype_profiles.csv
- question_cluster_summary.csv
- regression_test_results.csv
""")