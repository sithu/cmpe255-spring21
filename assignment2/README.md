 # Assignment 2

In this assignment, you will be building a SVM classifier to label human images.

## Dataset

You will use the labeled faces in the [Wild dataset](https://www.kaggle.com/c/labeled-faces-in-the-wild/overview) which consists of several thousand collated photos of the various public figures.

```python
from sklearn.datasets import fetch_lfw_people

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images_shape)
```

Each image contains [62x47] or nearly 3,000 pixels. Use each pixel value as a feature. You will use RandomizedPCA to extract 150 fundamental components to feed into your SVM model as a single pipeline.

```python
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
```

1. Split the data into a training and testing set.

2. Use a grid search cross-validation to explore combinations of parameters: C=margin hardness and gama=radial basis function kernel and determine the best model based on:
 * precision 
 * recall
 * f1-score
 * support

 3. Draw a confusion matrix between features.

