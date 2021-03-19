import pandas as pd
import category_encoders as ce

ids = [1, 2, 3, 4, 5, 6, 7]
# Categorical features
colors = ['Red', 'Green', 'Blue']

df = pd.DataFrame(list(zip(ids, colors)), columns=['Ids', 'Colors'])

# raw data
print(df.head())

# Label encoding (via OrdinalEncoder) for a large number of classes in a categorial feature.
encoder = ce.OrdinalEncoder(cols='Colors')
result = encoder.fit_transform(df)

print("Label encoding")
print(result)

# One-hot encoding uses binary buckets to repesent classes.
one_hot_enc = ce.OneHotEncoder(cols='Colors')
result = one_hot_enc.fit_transform(df)
print("One-hot encoding")
print(result)
