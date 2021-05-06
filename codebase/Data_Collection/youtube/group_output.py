import pandas as pd

df = pd.read_csv("results.csv")


df.groupby('VideoId')['text'].apply(' '.join).reset_index().to_csv("results_per_video_v1.csv")