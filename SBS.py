# Sequential Backward Selection - SBS

from sklearn.base import clone
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


if __name__ == '__main__':

    try:
        df_wine = pd.read_csv('wine.data', header=None)
        print('Data loaded')
    except:
        df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        df_wine.to_csv('wine.data', index=False, header=None)
        print('Data downloaded')

    df_wine.columns = ['Class label', 'Alcohol', 'Apple acid', 'Ash', 'Ash alkalinity', 'Magnesium', 'Fenols', 'Flavonoids', 'Non-flavonoid phenols', 'Proanthocyanidins', 'Color intensity', 'hue', 'Transmitation 280/315 nm', 'Proline']
    print('Class label: ', np.unique(df_wine['Class label']))

    print(df_wine.head())

    # Inputing missing data (mean value)
    # siir = SimpleImputer(missing_values=np.nan, strategy='mean')
    # siir = siir.fit(df_wine.values)
    # df_wine = siir.transform(df_wine.values)

    # Create train and test data sets

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    stdsc = StandardScaler()
    mms = MinMaxScaler()

    X_train_std = stdsc.fit_transform(X_train)
    X_train_mms = mms.fit_transform(X_train)

    X_test_std = stdsc.fit_transform(X_test)
    X_test_mms = mms.fit_transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)


    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    # efficiency plot
    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Efficiency')
    plt.xlabel('Number of attributes')
    plt.grid()
    plt.tight_layout()
    plt.show()

