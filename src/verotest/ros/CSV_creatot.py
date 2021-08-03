import os

import pandas as pd

df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                   'mask': ['red', 'purple'],
                   'weapon': ['sai', 'bo staff']})
print(df)
df.to_csv(r'C:\Users\VeronikaF\Documents\Robin4lemi\data_whatever.csv')
y = pd.read_csv(r'C:\Users\VeronikaF\Documents\Robin4lemi\data_whatever.csv')
print(y)

print(os.getcwd())
