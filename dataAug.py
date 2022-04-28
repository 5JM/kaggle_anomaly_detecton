import pandas as pd

train_y = pd.read_csv("./open/train_df.csv")
temp_y = train_y.copy()

for idx, row in temp_y.iterrows():
    if "good" not in row["label"]:
        for _ in range(5):
            train_y = pd.concat([train_y, temp_y.iloc[[idx]]])

train_y.to_csv('open/train_df_aug.csv', index=False)