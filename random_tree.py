# Evaluation the significance of attributes using the random forest algorithm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


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

    feat_labels = df_wine.columns[1:]

    forest = RandomForestClassifier(n_estimators=500, random_state=1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    plt.title('Feature materiality')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')

    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
    X_selected = sfm.transform(X_train)
    print('The number of features that meet a given threshold criterion:', X_selected.shape[1])

    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))