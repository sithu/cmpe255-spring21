import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "FIXME"

def save_fig(fig_id, tight_layout=True):
  # path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
  # path = os.path.join(fig_id + ".png")
  path = "five" + ".png"
  print("Saving figure", fig_id)
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format='png', dpi=300)

    

def random_digit(X):
    # return '1'
    # print('X keys maybe :',X)
    some_digit = X.iloc[36000]
    some_digit_image = some_digit.values.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.get_cmap('binary'),interpolation="nearest")
    plt.axis("off")

    save_fig("some_digit_plot")
    plt.show()
    return some_digit

   
def load_and_sort():
  try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml(name='mnist_784', cache=True)
    mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
    sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
  except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
  # mnist["data"], mnist["target"]
  return mnist["data"], mnist["target"]



def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    X_train = mnist.data.iloc[reorder_train]
    y_train = mnist.target.iloc[reorder_train]
    X_test = mnist.data.iloc[reorder_test + 60000]
    y_test = mnist.target.iloc[reorder_test + 60000]
    return X_train, y_train, X_test, y_test


def train_predict(some_digit, X_train, y_train):
    shuffle_index = np.random.permutation(60000)
    X_train, y_train =  X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

    # Example: Binary number 4 Classifier
    y_train_5 = (y_train == 5)
    # y_test_5 = (y_test == 5)

    from sklearn.linear_model import SGDClassifier
    SGD = SGDClassifier()
    SGD.fit(X_train, y_train_5)
    SGD.predict([some_digit])
    return SGD
    
def calculate_cross_val_score(classifier, y_train_5, X_train):
    from sklearn.model_selection import cross_val_score
    print(cross_val_score(classifier, X_train, cv=3, scoring="accuracy"))


X,y = load_and_sort()
some_digit = random_digit(X)
X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[60000:], y.iloc[:60000], y.iloc[60000:]
# print(len(mnist))
SGD = train_predict(some_digit, X_train, y_train)
calculate_cross_val_score(SGD, y_train, X_train)