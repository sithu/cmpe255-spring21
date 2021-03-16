import numpy as np
import os

np.random.seed(42)

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
    # print("Saving figure", fig_id)
    path = "five" + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()  
        plt.savefig(path, format='png', dpi=300)    

def random_digit():
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,
            interpolation="nearest")
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
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def train_predict(some_digit,X_train, X_test, y_train, y_test):
    import numpy as np
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # Example: Binary number 4 Classifier
    y_train_4 = (y_train == 4)
    y_test_4 = (y_test == 4)

    from sklearn.linear_model import SGDClassifier
    # TODO
    # print prediction result of the given input some_digit
    sgd = SGDClassifier(max_iter=10, random_state=43)
    model = sgd.fit(X_train, y_train_5)
    prediction = sgd.predict([some_digit])
    return model, X_train, y_train_5
    
    
def calculate_cross_val_score():
    # TODO
    from sklearn.model_selection import cross_val_score
    return(cross_val_score(model, X_train, y_train_5, cv=2, scoring="accuracy"))

if __name__ == "__main__":
    # execute only if run as a script
    X,y= load_and_sort()
    some_digit= random_digit(X)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    model, X_train, y_train_5=train_predict(some_digit, X_train, X_test, y_train, y_test)
    score=calculate_cross_val_score()
    print(score)