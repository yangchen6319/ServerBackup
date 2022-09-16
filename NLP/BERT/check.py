import pandas as pd

df = pd.read_csv('./data/test.txt', sep='\t', names=['label', 'review'])
counts = df['label'].value_counts()
print(type(counts))
print(counts)