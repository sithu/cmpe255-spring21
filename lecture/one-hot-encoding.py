import pandas as pd

ids = [1, 2, 3, 4, 5, 6, 7]
# Categorical features
colors = ['Red', 'Green', 'Blue']

df = pd.DataFrame(list(zip(ids, colors)), columns=['Ids', 'Colors'])

# raw data
print(df.head())

# Approach 1
# Pandas libaray's get_dummies()
# one-hot encoded vectors
y = pd.get_dummies(df.Colors, prefix='Color')
print(y.head())

# Approach 2
# Scikit-Learn OneHotEncoder and LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
y = LabelBinarizer().fit_transform(df.Colors)

print(y)

from sklearn.preprocessing import OneHotEncoder
x = [[1, "Red"], [2, "Green"], [3, "Blue"]]
y = OneHotEncoder().fit_transform(x).toarray()
print(y)

# Feature Cross between Colors and Sizes using PolynomialFeatures
x = [["M", "Red"], ["L", "Green"], ["XL", "Blue"]]
one_hot_enc = OneHotEncoder().fit_transform(x).toarray()
print(f"One-hot encoded {x}")
print(one_hot_enc)

# Approach 1 - Scikit Learn's PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
trans = PolynomialFeatures(degree=2)
data = trans.fit_transform(one_hot_enc)
print(data)

# Approach 2 - Tensorflow's feature_column
# import tensorflow as tf
# sizes = ['M', 'L', 'XL']
# colors_x_sizes_features = tf.feature_column.crossed_column(set([colors, sizes]), hash_bucket_size=1000)
# print(colors_x_sizes_features)