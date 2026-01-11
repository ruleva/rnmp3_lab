import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

# Find the CSV file
csv_files = glob.glob(r'C:\Users\rulev\PycharmProjects\rnmp3_lab\data\*.csv')
input_file = [f for f in csv_files if 'diabetes' in f.lower()][0]

# Read and split
df = pd.read_csv(input_file)
offline_df, online_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Diabetes_binary'])

# Save
offline_df.to_csv(r'C:\Users\rulev\PycharmProjects\rnmp3_lab\data\offline.csv', index=False)
online_df.to_csv(r'C:\Users\rulev\PycharmProjects\rnmp3_lab\data\online.csv', index=False)

print(f"✓ Split complete: {len(offline_df)} offline, {len(online_df)} online")