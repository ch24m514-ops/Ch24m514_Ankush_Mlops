import os, pandas as pd

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
os.makedirs("data/raw", exist_ok=True)
out = "data/raw/titanic.csv"
print(f"Downloading Titanic dataset to {out} ...")
df = pd.read_csv(URL)
df.to_csv(out, index=False)
print("Done.")
