import pandas as pd
from collections import defaultdict

df = pd.read_csv('ratings-ordered.csv')
print(df.head())
n_users = df.user.nunique()

subset = df.query('item < 100')
subset['label'] = subset['rating'].map({
    'willsee': 1,
    'wontsee': 1,
    'dislike': 2,
    'neutral': 2,
    'like': 2,
    'favorite': 2
})
subset[['user', 'item', 'label']].to_csv('ratings-subset.csv', index=False)

labels = defaultdict(lambda: 0)
for user, item, label in zip(subset['user'], subset['item'], subset['label']):
    labels[user, item] = label
full = []
for user in range(n_users):
    for item in range(100):
        full.append({
            'user': user,
            'item': item,
            'label': labels[user, item]
        })
pd.DataFrame(full).to_csv('ratings-full.csv', index=False)
