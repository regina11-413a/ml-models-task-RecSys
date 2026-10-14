import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

events = pd.read_csv("data/events_filtered.csv")
items = pd.read_csv("data/item_features.csv")

items["categoryid"] = items["categoryid"].fillna("unknown").astype(str)
items["brand"] = items["brand"].fillna("unknown").astype(str)
items["features"] = items["categoryid"] + " " + items["brand"]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(items["features"])

item_ids = items["itemid"].astype(int).tolist()
item_id_to_idx = {item: idx for idx, item in enumerate(item_ids)}
idx_to_item_id = {idx: item for idx, item in enumerate(item_ids)}

def recommend_similar(item_id, n=10):
    item_id = int(item_id)
    if item_id not in item_id_to_idx:
        return None
    idx = item_id_to_idx[item_id]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]
    return [idx_to_item_id[int(i)] for i in similar_idx if int(i) in idx_to_item_id]

if __name__ == "__main__":
    items_with_features = set(item_ids)
    events_with_features = events[events["itemid"].isin(items_with_features)]

    if len(events_with_features) == 0:
        print("Ни один товар из events_filtered не имеет признаков.")
    else:
        sample_item = int(events_with_features["itemid"].iloc[0])
        print(f"Товар: {sample_item}")
        print("Похожие товары:")
        result = recommend_similar(sample_item, 10)
        print(result)