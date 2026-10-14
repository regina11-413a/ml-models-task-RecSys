import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

events = pd.read_csv("data/events_filtered.csv")
items = pd.read_csv("data/item_features.csv")

items = items[items["itemid"].isin(events["itemid"].unique())].copy()
items = items.drop_duplicates(subset="itemid")

events = events.sort_values("timestamp")
split_idx = int(len(events) * 0.8)
train = events.iloc[:split_idx]
test = events.iloc[split_idx:]
test = test[test["visitorid"].isin(train["visitorid"].unique())]
test_users = test["visitorid"].unique()

print(f"Train: {len(train)} событий, Test: {len(test)} событий")
print(f"Товаров с признаками: {len(items)}")

items["categoryid"] = items["categoryid"].fillna("unknown").astype(str)
items["brand"] = items["brand"].fillna("unknown").astype(str)
items["features"] = items["categoryid"] + " " + items["brand"]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(items["features"])
item_ids = items["itemid"].astype(int).tolist()
item_id_to_idx = {item: idx for idx, item in enumerate(item_ids)}
idx_to_item_id = {idx: item for idx, item in enumerate(item_ids)}


def precision_at_k(recs, relevant, k):
    rec_k = recs[:k]
    return len(set(rec_k) & set(relevant)) / k if k > 0 else 0.0


def recall_at_k(recs, relevant, k):
    rec_k = recs[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recs, relevant, k):
    rec_k = recs[:k]
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


popular_items = (
    train[train["event"] == "transaction"]
    .groupby("itemid").size()
    .sort_values(ascending=False)
    .index.tolist()
)


def content_based_recs(user_id, n=10):
    user_items = train[train["visitorid"] == user_id]["itemid"].tolist()
    user_items = [i for i in user_items if i in item_id_to_idx]
    if not user_items:
        return []
    idxs = [item_id_to_idx[i] for i in user_items]
    user_profile = tfidf_matrix[idxs].mean(axis=0)
    user_profile = np.asarray(user_profile)
    sim = cosine_similarity(user_profile, tfidf_matrix).flatten()
    top = np.argsort(sim)[::-1]
    recs = [idx_to_item_id[int(i)] for i in top if int(i) in idx_to_item_id]
    recs = [r for r in recs if r not in user_items]
    return recs[:n]


user_codes = train["visitorid"].astype("category").cat.codes.values
item_codes = train["itemid"].astype("category").cat.codes.values
matrix = csr_matrix(
    (np.ones(len(train)), (user_codes, item_codes)),
    shape=(user_codes.max() + 1, item_codes.max() + 1)
)
item_categories = train["itemid"].astype("category").cat.categories
code_to_item = {code: int(item) for code, item in enumerate(item_categories)}
item_to_code = {item: code for code, item in code_to_item.items()}


def collaborative_recs(user_id, n=10):
    user_events = train[train["visitorid"] == user_id]["itemid"].tolist()
    user_events = [i for i in user_events if i in item_to_code]
    if not user_events:
        return []
    scores = np.zeros(matrix.shape[1])
    for item in user_events:
        code = item_to_code[item]
        item_vector = matrix.T[code]
        sim = cosine_similarity(item_vector, matrix.T).flatten()
        scores += sim
    top = scores.argsort()[::-1]
    recs = [code_to_item[int(c)] for c in top if int(c) in code_to_item]
    recs = [r for r in recs if r not in user_events]
    return recs[:n]


K = 10
results = {"popular": [], "content": [], "collab": []}
sample_users = test_users[:200]

for user in sample_users:
    relevant = test[test["visitorid"] == user]["itemid"].tolist()
    if not relevant:
        continue

    recs_pop = popular_items[:K]
    recs_content = content_based_recs(user, K)
    recs_collab = collaborative_recs(user, K)

    for name, recs in [("popular", recs_pop), ("content", recs_content), ("collab", recs_collab)]:
        results[name].append({
            "precision": precision_at_k(recs, relevant, K),
            "recall": recall_at_k(recs, relevant, K),
            "ndcg": ndcg_at_k(recs, relevant, K)
        })

print(f"\nМетрики при K={K} (усреднено по {len(sample_users)} пользователям):")
print(f"{'Подход':<12} {'Precision@K':<14} {'Recall@K':<12} {'NDCG@K':<10}")
print("-" * 50)
for name in ["popular", "content", "collab"]:
    df = pd.DataFrame(results[name])
    print(f"{name:<12} {df['precision'].mean():<14.4f} {df['recall'].mean():<12.4f} {df['ndcg'].mean():<10.4f}")