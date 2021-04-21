 # Assignment 2

In this assignment, you will be building a SVM classifier to label [famous people's images](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py).

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

## Requirements

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

2. Use a [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) cross-validation to explore combinations of [parameters](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to determine the best model: 
   - C: margin hardness E.g. [1, 5, 10, 50]
   - gama: radial basis function kernel E.g. [0.0001, 0.0005, 0.001, 0.005]
 * precision 
 * recall
 * f1-score
 * support

 3. Draw a 4x6 subplots of images using names as label with color black for correct instances and red for incorrect instances.

 4. Draw a confusion matrix between features in a [heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) with X-axis of 'Actual' and Y-axis of 'Predicted'.

