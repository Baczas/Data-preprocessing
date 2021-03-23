import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
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

    # Accuracy and precision
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
    lr.fit(X_train_std, y_train)
    print('Train data accuracy: ', lr.score(X_train_std, y_train))
    print('Test data accuracy: ', lr.score(X_test_std, y_test))
    print('Intercepts: ', lr.intercept_)
    print('Weights coefficients: ', lr.coef_)

    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
    weights, params = [], []
    for c in np.arange(-4., 6.):
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=10.**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('Weight coefficient')
    plt.xlabel('C - Inverse of regularization strength')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()




