import pandas as pd

products = pd.read_csv("data/product.csv", sep="\t")

print(products.shape)
print(products.columns)
print(products.head())
print(products.isna().sum())