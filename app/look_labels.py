import pandas as pd

queries = pd.read_csv("data/query.csv", sep="\t")
labels = pd.read_csv("data/label.csv", sep="\t")

print(queries.shape)
print(queries.head())
print()
print(labels.shape)
print(labels.head())
print()
print(labels["label"].value_counts())