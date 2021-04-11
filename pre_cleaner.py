from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def clean_decisions(data, col, percent, switch=False, chart=True):
    if chart:
        if switch:
            plot(data, col, percent, size=(2, 1))
        else:
            plot(data, col, percent)
    print('What to do? (d: drop column, s: skip (correct later manually), f: fill with most_frequent value')
    print('me: filled with mean value, oh: One Hot encoding, bin: Binary encoding')
    answer = input()

    if answer == 'd':  # Drop column
        data = data.drop(col, axis=1)
        print('d: Column dropped')

    elif answer == 's':  # skip and do nothing
        print('s: Column skipped')

    elif answer == 'f':  # fill with most_frequent value
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        data[col] = imp.fit_transform(pd.DataFrame(data[col]))
        print('f: filled with most_frequent value')

    elif answer == 'me':  # fill with mean value
        if data[col].dtypes == np.int32 or data[col].dtypes == np.float32 or data[col].dtypes == np.int64 or data[col].dtypes == np.float64:
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            data[col] = imp.fit_transform(pd.DataFrame(data[col]))
            print('me: filled with mean value')
        else:
            print('Can\'t Impute data (non numerical values)')
            print('me: filled with mean value - FAILED!')

    elif answer == 'oh':  # One Hot encoding
        print('oh: One Hot encoding')

    elif answer == 'bin':  # binary encoding
        print('bin: Binary encoding')

    return data


def pre_cleaner(data):
    columns = data.columns
    for col in data.columns:
        percent = data[col].isna().sum() / data[col].shape[0]
        print(f'\nColumn: {col}')
        print('Number of unique values:', data[col].nunique(), '; Percent of missing data:', percent, '%; Column data:', data[col].dtypes)
        print(data[col].describe())
        data_type = (data[col].dtypes == np.int32 or
                     data[col].dtypes == np.float32 or
                     data[col].dtypes == np.int64 or
                     data[col].dtypes == np.float64)

        if data[col].isna().sum() == 0 and data_type:
            print('Data are correct in this column.')
        elif data[col].isna().sum() != 0 and data_type:
            print('Data are correct format but have empty spots.')
            print('Fill with mean value, pass, drop column, or do what you want')
            data = clean_decisions(data, col, percent, chart=False)
        else:
            if data[col].nunique() > 10:
                while True:
                    ans = input(f'Many unique values! ({data[col].nunique()}) Do you want to print data on the graph?')
                    if ans == ('y' or 'Y'):
                        data = clean_decisions(data, col, percent)
                        break
                    elif ans == ('n' or 'N'):
                        print('n/N: Column skipped')
                        break
            else:
                data = clean_decisions(data, col, percent)

    data.to_csv('pre_cleaned_data.csv')
    print('Data exported to file. (pre_cleaned_data.csv)')


def plot(data, col, percent, size=(1, 2)):
    label = data[col].value_counts().index
    fig, (ax1, ax2) = plt.subplots(size[0], size[1])
    ax1.bar(label, data[col].value_counts().values)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    ax2.pie(data[col].value_counts().values, labels=label, autopct='%1.1f%%')
    fig.suptitle(f'Column: \'{col}\'   -   Missing data: {percent:.2f}% ({data[col].isna().sum()})', fontsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load example data:  https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists
    try:
        FILE_NAME = 'aug_train.csv'
        df = pd.read_csv(FILE_NAME)
        df = df.iloc[:1000, :]
        print(f'Data loaded from file: \"{FILE_NAME}\"\n')
        print(df.columns)
        print(f'\nNumber of columns: {df.shape[1]}')
        input('Wait for key to start')

        pre_cleaner(df)  # it export cleaned data automatically
    except:
        print('You don\'t have data file.')
