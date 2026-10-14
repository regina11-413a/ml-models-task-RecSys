import pandas as pd

items = pd.read_csv("data/item_properties_part1.csv")
category_tree = pd.read_csv("data/category_tree.csv")

category = items[items["property"] == "categoryid"][["itemid", "value"]].copy()
category.columns = ["itemid", "categoryid"]

brand = items[items["property"] == "790"][["itemid", "value"]].copy()
brand.columns = ["itemid", "brand"]

item_features = category.merge(brand, on="itemid", how="outer")

item_features.to_csv("data/item_features.csv", index=False)

print(f"Товаров с признаками: {len(item_features)}")
print(item_features.head())