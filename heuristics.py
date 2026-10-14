import pandas as pd

events = pd.read_csv("data/events_filtered.csv")

popular = (
    events[events["event"] == "transaction"]
    .groupby("itemid")
    .size()
    .sort_values(ascending=False)
    .head(100)
)

print("Топ-10 популярных товаров (по покупкам):")
print(popular.head(10))

def recommend_popular(n=10):
    return popular.head(n).index.tolist()

if __name__ == "__main__":
    print("\nРекомендация для любого пользователя:")
    print(recommend_popular(10))