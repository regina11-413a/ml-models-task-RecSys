import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

events = pd.read_csv("data/events_filtered.csv")

user_codes = events["visitorid"].astype("category").cat.codes.values
item_codes = events["itemid"].astype("category").cat.codes.values

matrix = csr_matrix(
    (np.ones(len(events)), (user_codes, item_codes)),
    shape=(user_codes.max() + 1, item_codes.max() + 1)
)

item_categories = events["itemid"].astype("category").cat.categories
code_to_item = {code: int(item) for code, item in enumerate(item_categories)}
item_to_code = {item: code for code, item in code_to_item.items()}

def recommend_similar(item_id, n=10):
    item_id = int(item_id)
    if item_id not in item_to_code:
        return None
    code = item_to_code[item_id]
    item_vector = matrix.T[code]
    sim_scores = cosine_similarity(item_vector, matrix.T).flatten()
    similar_codes = sim_scores.argsort()[::-1][1:n+1]
    return [code_to_item[int(c)] for c in similar_codes if int(c) in code_to_item]

if __name__ == "__main__":
    sample_item = int(events["itemid"].iloc[0])
    print(f"Товар: {sample_item}")
    print("Похожие товары (item-based CF):")
    result = recommend_similar(sample_item, 10)
    print(result)