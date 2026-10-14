import pandas as pd

events = pd.read_csv("data/events.csv")

user_counts = events["visitorid"].value_counts()
item_counts = events["itemid"].value_counts()

active_users = user_counts[user_counts >= 5].index
popular_items = item_counts[item_counts >= 5].index

filtered = events[
    events["visitorid"].isin(active_users) &
    events["itemid"].isin(popular_items)
].copy()

print(f"Было: {len(events)} событий, {events['visitorid'].nunique()} пользователей, {events['itemid'].nunique()} товаров")
print(f"Стало: {len(filtered)} событий, {filtered['visitorid'].nunique()} пользователей, {filtered['itemid'].nunique()} товаров")

filtered.to_csv("data/events_filtered.csv", index=False)