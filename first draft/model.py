import duckdb
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import ast
import random
import implicit
import math

db = duckdb.connect(":memory:")
db.sql(f"CREATE TABLE novel AS SELECT * FROM '../eda/data/novels_0.1.5.csv';")

# Eval Functions

def precision_at_k(recommended, relevant, k):
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / k

def recall_at_k(recommended, relevant, k):
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / len(relevant)

def average_precision_at_k(recommended, relevant, k):
    ap, hits = 0.0, 0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)

def print_metrics(dict_truth, dict_recommend, model, K):
    precision_list = []
    recall_list = []
    ap_list = []

    for user, relevant in dict_truth.items():
        recommended = dict_recommend[user]

        precision_list.append(precision_at_k(recommended, relevant, K))
        recall_list.append(recall_at_k(recommended, relevant, K))
        ap_list.append(average_precision_at_k(recommended, relevant, K))

    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mean_ap = sum(ap_list) / len(ap_list)

    print(f"Random {model} Precision@{K}: {mean_precision:.6f}")
    print(f"Random {model} Recall@{K}: {mean_recall:.6f}")
    print(f"Random {model} MAP@{K}: {mean_ap:.6f}")
    return (mean_precision, mean_recall, mean_ap)

ids = db.sql("SELECT id FROM novel").fetchall()
novel_id_to_index = {}
for (i, id) in enumerate(ids):
    novel_id_to_index[id] = i

rec_lists = db.sql("SELECT id, recommendation_list_ids, genres, tags FROM novel").fetchall()

rec_dict = defaultdict(list)
rec_genre = defaultdict(list)
rec_tag = defaultdict(list)
for rec_list in rec_lists:
    ids = rec_list[1]
    if ids is not None:
        split_ids = ids.split(",")
        split_genres = rec_list[2].split(",")
        clean_genres = []
        for genre in split_genres:
            clean_genre = genre.replace("[", "").replace("]","")
            if clean_genre:
                clean_genres.append(clean_genre)
        split_tags = rec_list[3].split(",")
        clean_tags = []
        for tag in split_tags:
            clean_tag = tag.replace("[", "").replace("]","")
            if clean_tag:
                clean_tags.append(clean_tag)
        for id in split_ids:
            clean_id = id.replace("[", "").replace("]","").replace(" ", "")
            rec_dict[clean_id].append(rec_list[0])
            rec_genre[clean_id] = clean_genres
            rec_tag[clean_id] = clean_tags

def normalize_text(text):
    return text.lower().replace(" ", "").replace("-", "").replace("*", "").replace("/", "").strip().strip("'")

rec_lists = db.sql("SELECT id, recommendation_list_ids, genres, tags FROM novel").fetchall()
novel_genre = defaultdict(list)
novel_tag = defaultdict(list)

for novel in rec_lists:
    novel_id = novel[0]
    split_genres = novel[2].split(",")
    for genre in split_genres:
        clean_genre = genre.replace("[", "").replace("]","")
        if clean_genre:
            novel_genre[novel_id].append(clean_genre)
    split_tags = novel[3].split(",")
    for tag in split_tags:
        clean_tag = tag.replace("[", "").replace("]","")
        if clean_tag:
            novel_tag[novel_id].append(clean_tag)

cleaned_novel_tag = {
    novel: [normalize_text(t) for t in tags]
    for novel, tags in novel_tag.items()
}

cleaned_novel_genre = {
    novel: [normalize_text(g) for g in genres]
    for novel, genres in novel_genre.items()
}

novel_tag_vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None
)

novel_genre_vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None
)

tfidf_novel_tag_matrix = novel_tag_vectorizer.fit_transform(cleaned_novel_tag.values())

tfidf_novel_genre_matrix = novel_genre_vectorizer.fit_transform(cleaned_novel_genre.values())

df_tfidf_novel_tag = pd.DataFrame(
    tfidf_novel_tag_matrix.toarray(),
    index=cleaned_novel_tag.keys(),
    columns=novel_tag_vectorizer.get_feature_names_out()
)

df_tfidf_novel_genre = pd.DataFrame(
    tfidf_novel_genre_matrix.toarray(),
    index=cleaned_novel_genre.keys(),
    columns=novel_genre_vectorizer.get_feature_names_out()
)

df_tfidf_novel_all = pd.concat(
    [df_tfidf_novel_tag, df_tfidf_novel_genre],
    axis=1
).fillna(0)

def recommend_similar_novels(novel_id, top_k=5):
    target_vec = df_tfidf_novel_all.loc[[novel_id]]

    sims = cosine_similarity(target_vec, df_tfidf_novel_all)[0]

    sim_series = pd.Series(sims, index=df_tfidf_novel_all.index)

    sim_series = sim_series.drop(novel_id)
    return sim_series.sort_values(ascending=False).head(top_k)

max_v = 0
total = 0
min_v = 999999

for rec in rec_dict.items():
    length = len(rec[1])
    total += length
    if (length > max_v):
        max_v = length
    if (length < min_v):
        min_v = length

print(max_v, total/len(rec_dict.items()), min_v)

rec_list_df = pd.DataFrame([
    {"rec_id": k, "novel_id": v}
    for k, values in rec_dict.items()
    for v in values
])
print(rec_list_df)

train_rows = []
test_rows = []

for user, group in rec_list_df.groupby("rec_id"):
    if len(group) < 2:
        continue
    loo = LeaveOneOut()
    X = group["novel_id"].values.reshape(-1, 1) 
    for train_idx, test_idx in loo.split(X):
        train_rows.append(group.iloc[train_idx])
        test_rows.append(group.iloc[test_idx])
        break 

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df = pd.concat(test_rows).reset_index(drop=True)

print("Train set size:", len(train_df))
print("Test set size:", len(test_df))


# Random Baseline
def random_rec(novel_dict):
    return random.choice(list(novel_dict.keys()))[0]

train_df["user_idx"] = train_df["rec_id"].astype("category").cat.codes
train_df["item_idx"] = train_df["novel_id"].astype("category").cat.codes

interaction_matrix = coo_matrix(
    ([1]*len(train_df), (train_df["user_idx"], train_df["item_idx"]))
).tocsr()

# Initialize ALS model
cf_model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)

# implicit expects item-user matrix
cf_model.fit(interaction_matrix.T)  # transpose: items x users

#TF-IDF Content Based Model
tfidf_matrix = df_tfidf_novel_all.values
novel_ids = df_tfidf_novel_all.index.to_list()
novel_id_to_idx = {nid: i for i, nid in enumerate(novel_ids)}
idx_to_novel_id = {i: nid for i, nid in enumerate(novel_ids)}

def get_similar_novels(novel_id, top_k=5):
    idx = novel_id_to_idx[novel_id]
    vec = tfidf_matrix[idx].reshape(1, -1)
    sims = cosine_similarity(vec, tfidf_matrix)[0]

    sims[idx] = -1

    top_idxs = sims.argsort()[::-1][:top_k]
    return [idx_to_novel_id[i] for i in top_idxs], sims[top_idxs]

def get_user_profile(novel_list):
    idxs = [novel_id_to_idx[n] for n in novel_list if n in novel_id_to_idx]
    if not idxs:
        return None
    return tfidf_matrix[idxs].mean(axis=0)

def recommend_for_user(user_novels, K=5):
    profile = get_user_profile(user_novels)
    if profile is None:
        return []

    sims = cosine_similarity(profile.reshape(1, -1), tfidf_matrix)[0]

    for n in user_novels:
        if n in novel_id_to_idx:
            sims[novel_id_to_idx[n]] = -1

    top_idxs = sims.argsort()[::-1][:K]
    return [idx_to_novel_id[i] for i in top_idxs]

# Fetch the raw data
novel_data = db.sql("""
    SELECT 
        id AS novel_id,
        original_language,
        genres,
        on_reading_lists
    FROM novel
""").fetchall()

df_novel = pd.DataFrame(novel_data, columns=['novel_id', 'original_language', 'genres', 'on_reading_lists'])

# Convert genres from string to list
def parse_list_column(x):
    if x:
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return parsed[0]
            return parsed
        except:
            return []
    return []

df_novel['genres'] = df_novel['genres'].apply(parse_list_column)

df_novel['on_reading_lists'] = df_novel['on_reading_lists'].astype(int)

df_novel['original_language'] = df_novel['original_language'].str.lower().str.strip()

df_novel.head()

def get_favorite_language(user_novels, novel_df):
    langs = novel_df.loc[novel_df['novel_id'].isin(user_novels), 'original_language']
    most_common_lang = langs.mode()
    if len(most_common_lang) > 0:
        return most_common_lang[0]
    return None

def filter_candidates(novel_df, language, user_genres=None, exclude_novels=[]):
    candidates = novel_df[novel_df['original_language'] == language].copy()

    if user_genres is not None:
        candidates = candidates[candidates['genres'].apply(lambda g: any(genre in g for genre in user_genres))]

    if exclude_novels:
        candidates = candidates[~candidates['novel_id'].isin(exclude_novels)]

    return candidates

def rank_by_popularity(candidates, top_k=10):
    return candidates.sort_values('on_reading_lists', ascending=False).head(top_k)

def recommend_popular_for_user(user_novels, novel_df, top_k=10):
    langs = novel_df.loc[novel_df['novel_id'].isin(user_novels), 'original_language']
    if len(langs) == 0:
        fav_lang = None
    else:
        fav_lang = langs.mode()[0]

    genres_lists = novel_df.loc[novel_df['novel_id'].isin(user_novels), 'genres']
    user_genres = []
    for glist in genres_lists:
        if isinstance(glist, list):
            user_genres.extend(glist)
    user_genres = list(set(user_genres))

    if fav_lang is None:
        candidates = novel_df.copy()
    else:
        candidates = novel_df[novel_df['original_language'] == fav_lang].copy()
        if user_genres:
            candidates = candidates[candidates['genres'].apply(lambda g: any(genre in g for genre in user_genres))]
        candidates = candidates[~candidates['novel_id'].isin(user_novels)]

    recommended = candidates.sort_values('on_reading_lists', ascending=False).head(top_k)
    return recommended['novel_id'].tolist()
K = 5

user_test_dict = defaultdict(list)
for _, row in test_df.iterrows():
    user_test_dict[row["rec_id"]].append(row["novel_id"])

# Random baseline recommendations
user_random_recs = dict()

for user in user_test_dict:
    recs = []
    while len(recs) < K:
        rec = random_rec(novel_id_to_index)
        if rec not in recs:
            recs.append(rec)
    user_random_recs[user] = recs

user_cf_recs = {}

# Mapping from item_idx → novel_id
idx_to_novel = dict(enumerate(train_df["novel_id"].astype("category").cat.categories))

for user in user_test_dict:
    # Some test users may not exist in training set → skip
    if user not in train_df["rec_id"].values:
        user_cf_recs[user] = []
        continue

    user_idx = train_df.loc[train_df["rec_id"] == user, "user_idx"].iloc[0]

    recommended, _ = cf_model.recommend(
        user_idx,
        interaction_matrix,
        N=K,
        filter_already_liked_items=False
    )

    user_cf_recs[user] = [idx_to_novel[i] for i in recommended.astype(int)]

user_content_recs = {}
for user, novels in user_test_dict.items():
    user_content_recs[user] = recommend_for_user(novels, K=K)

user_pop_recs = {}
for user, novels in user_test_dict.items():
    user_pop_recs[user] = recommend_popular_for_user(novels, df_novel, top_k=K)

models = {
    "Random Baseline": user_random_recs,
    "CF Model": user_cf_recs,
    "Content TF-IDF": user_content_recs,
    "Popularity Heuristic": user_pop_recs
}

metrics_data = {}

for name, recs in models.items():
    mean_precision, mean_recall, mean_ap = print_metrics(user_test_dict, recs, name, K)
    metrics_data[name] = {
        "Precision@{}".format(K): mean_precision,
        "Recall@{}".format(K): mean_recall,
        "MAP@{}".format(K): mean_ap
    }

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_data).T

print(metrics_df)

# Plot
metrics_df.plot(kind="bar", figsize=(10,6))
plt.ylabel("Metric Value")
plt.title("Top-K Metrics Comparison")
plt.xticks(rotation=0)
plt.savefig("topk_metrics.png")
plt.show()

user_lengths = [len(v) for v in rec_dict.values()]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear scale
axes[0].hist(user_lengths, bins=range(1, 100), edgecolor="black")
axes[0].set_xlabel("Number of novels in user list")
axes[0].set_ylabel("Number of users")
axes[0].set_title("Linear Scale")

# Log scale
axes[1].hist(user_lengths, bins=range(1, 100), edgecolor="black")
axes[1].set_xlabel("Number of novels in user list")
axes[1].set_ylabel("Number of users")
axes[1].set_title("Log-Log Scale")
axes[1].set_xscale("log")
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("num_users_vs_num_novels_in_list.png")
plt.show()
