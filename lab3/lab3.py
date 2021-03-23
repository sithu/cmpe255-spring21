import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self):
        feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age',]
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train_test_split(self):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        return X_train, y_train

    def predict(self,model):
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
    def descision_tree_classifier(self,X_train,y_train):
        from sklearn import tree
        model = tree.DecisionTreeClassifier(random_state=0, max_depth=5)
        model.fit(X_train,y_train)
        return model

    def logisticregression(self,X_train,y_train):
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        return model
    
    def svm(self,X_train,y_train):
        from sklearn import svm
        model = svm.SVC(kernel='rbf')
        model.fit(X_train, y_train, )
        return model

if __name__ == "__main__":
    classifer = DiabetesClassifier()
    X_train, y_train =  classifer.train_test_split()
    model_regression = classifer.logisticregression(X_train,y_train)
    result = classifer.predict(model_regression)
    score = classifer.calculate_accuracy(result)
    print(f"logistic Regression: score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"Logistic Regression: confusion_matrix=${con_matrix}")

    model_svm = classifer.svm(X_train,y_train)
    result = classifer.predict(model_svm)
    score = classifer.calculate_accuracy(result)
    print(f"Support Vector Machine: score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"Support Vector Machine: confusion_matrix=${con_matrix}")


    model_descision_tree= classifer.descision_tree_classifier(X_train,y_train)
    result = classifer.predict(model_descision_tree)
    score = classifer.calculate_accuracy(result)
    print(f"Descision Tree: score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"Descision Tree: confusion_matrix=${con_matrix}")
    
