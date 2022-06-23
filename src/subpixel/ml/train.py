import sklearn
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Regression:
    def __init__(self, df, target_col, type=None):

        self.df = df
        self.target_col = target_col
        self.X = self.df[self.df.columns.difference([self.target_col])]
        self.y = self.df[self.target_col]
        self.type = type

        self.model_dict = {
            "Nearest Neighbors": KNeighborsClassifier(3),
            "Linear SVM": SVC(kernel="linear", C=0.025),
            "RBF SVM": SVC(gamma=2, C=1),
            "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1
            ),
            "Neural Net": MLPClassifier(alpha=1),
            "AdaBoost": AdaBoostClassifier(),
            "Naive Bayes": GaussianNB(),
            "QDA": QuadraticDiscriminantAnalysis(),
        }

        if self.type == "linear":
            self.model = sklearn.linear_model.LinearRegression()
        else:
            self.model = self.model_dict[self.find_classfier()[0]]

        self.model.fit(self.X, self.y)

    def find_classfier(self):

        _THRESHOLD = 500

        if len(self.df) > _THRESHOLD:
            self._X = resample(self.X, replace=False, n_samples=_THRESHOLD)
            self._y = resample(self.y, replace=False, n_samples=_THRESHOLD)

        else:
            self._X = self.X
            self._y = self.y

        model_scores = {}

        for name, model in zip(self.model_dict.keys(), self.model_dict.values()):
            model.fit(self._X, self._y)
            model_scores[name] = model.score(self._X, self._y)

        model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1]))

        return list(model_scores.keys())[0], model_scores

    def predict(self, df):
        return self.model.predict(df)

    def score(self):
        return self.model.score(self.X, self.y)

    def score_with_test(self, X_test, y_test):
        return self.model.score(X_test, y_test)
