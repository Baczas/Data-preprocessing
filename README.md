# Data-preprocessing

### pre_cleaner.py - Pre cleaning data (by column)

## pre_cleaner(data)
data -> pd.DataFrame

The pre_cleaner() function visualizes the data from a given column and then asks what to do with the column:
- d: drop column
- s: skip (correct later manually)
- f: fill with most_frequent value
- me: filled with mean value
- oh: One Hot encoding
- bin: Binary encoding
##### You can execute above commands in cascade by writing them after comma i.e. f,oh

## to_clean(data)
data -> pd.DataFrame

Function return name of columns which need to be clean
##### Chcecking data standarization/normalization will be implemented
