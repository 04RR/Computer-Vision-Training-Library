from statistics import median
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd


def accuracy(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def correlation_matrix(df, cols=False):

    if cols:
        df = df[cols]

    return df.corr()


def find_outliers(df, cols=False, remove=False):

    if cols:
        df = df[cols]
        numeric_cols = df._get_numeric_data().columns.tolist()
    else:
        numeric_cols = df._get_numeric_data().columns.tolist()

    outliers = {}

    for col in numeric_cols:

        outlier_list = []

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        low_bound = q1 - (iqr * 1.5)
        high_bound = q3 + (iqr * 1.5)

        for i, val in enumerate(df[col]):
            if val < low_bound or val > high_bound:
                outlier_list.append(i)

                if remove:
                    df.drop(df.index[i], inplace=True)

        outliers[col] = outlier_list

    return outliers


def boxplot(df, cols=False):

    if cols:
        df = df[cols]
        numeric_cols = df._get_numeric_data().columns.tolist()
    else:
        numeric_cols = df._get_numeric_data().columns.tolist()

    i = 1
    plt.figure(figsize=(15, 25))
    for col in numeric_cols:
        plt.subplot(6, 3, i)
        sns.boxplot(y=df[col], color="green")
        i += 1

    plt.show()


def get_combinations(list_of_values):
    return list(itertools.combinations(list_of_values, 2))


# NEEDS TO BE CHANGED TO DISPLAY IN SAME PAGE
def feature_correlation(df, cols=False, kind="reg"):

    if cols:
        lst = get_combinations(cols)

    else:
        lst = get_combinations(df.columns)

    for i, j in lst:

        sns.jointplot(x=i, y=j, data=df, kind=kind, truncate=False, color="m", height=7)

    plt.show()


def fill_nan_with_mean(df):
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def delete_row_with_nan(df):
    df.dropna(inplace=True)
    return df


def pie_chart(df, col):

    df[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.show()


def count_plot(df, col):

    df[col].value_counts().plot(kind="bar")
    plt.show()


def feature_importance(x, y, show_plot=False):
    model = ExtraTreesClassifier()
    model.fit(x, y)

    feat_importances = pd.Series(model.feature_importances_, index=x.columns)

    if show_plot:
        feat_importances.nlargest(12).plot(kind="barh")
        plt.show()

    return feat_importances


def histogram(df, cols, bins=10):

    n = len(cols)

    plt.figure(figsize=(10, 10))

    for i, col in enumerate(cols):
        plt.subplot(n, 1, i + 1)
        sns.histplot(
            df[col],
            bins=bins,
            color="Red",
            kde_kws={"color": "y", "lw": 3, "label": "KDE"},
        )

    plt.show()


def get_median(df, col):
    return df[col].median()


def get_mean(df, col):
    return df[col].mean()


def check_for_outliers(df, cols=False, threshold=10):

    cols = df.columns if cols is False else cols

    cols_with_outliers = []

    for col in cols:
        mean = get_mean(df, col)
        median = get_median(df, col)

        if abs(mean - median) > (threshold / 100) * max(mean, median):
            cols_with_outliers.append(col)

    return cols_with_outliers


def get_correlation_with_target(df, target, cols=False):

    if cols:
        df = df[cols]

    return df.corrwith(target).sort_values(ascending=False)


def get_kurtosis(df, col):
    return df[col].kurtosis()
