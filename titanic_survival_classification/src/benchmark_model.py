import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
plt.style.use('ggplot')


def format_data(df, title_options=set(['Capt.','Col.','Don.','Dr.','Major.','Master.','Miss.','Mr.','Mrs.','Ms.','Rev.'])):
    '''
    Converts the Pclass, SibSp and Parch columns to be int types, as well
    as creates a new column, "Title", that is the prefix associated with
    each passenger (i.e. Mr., Mrs., Master., etc.)

    Parameters:
    ----------
    df : (Pandas DataFrame)
        Must contain columns Pclass, SibSp, Parch and Name.
    title_options : (set)
        A set of possible strings that a passenger could have been called.
        (i.e. Master., Dr., Miss., etc.)

    Returns:
    ----------
    None
    '''
    df['Pclass'] = df['Pclass'].astype(int)
    df['SibSp'] = df['SibSp'].astype(int)
    df['Parch'] = df['Parch'].astype(int)
    df['Split'] = df['Name'].apply(lambda x: x.split())
    df['Title'] = df['Split'].apply(lambda x: title_options.intersection(x).pop() if len(title_options.intersection(x)) > 0 else 'Misc.')
    df['Title'] = df['Title'].astype(object)
    df.drop('Split', axis=1, inplace=True)


def impute_age(df):
    '''
    Imputes the mean age based on subgroups defined by all possible
    combinations of the unique values in the Sex, Pclass and Title
    columns.

    Parameters:
    ----------
    df : (Pandas DataFrame)
        Must include columns "Title", "Pclass", "Sex" and "Age"

    Returns:
    ----------
    None
    '''
    check = set()
    for i in np.unique(df['Title']):
        mask = df['Title'] == i
        percent_nan = count_nans(df[mask], ['Age'], verbose=False)[0][1]
        if percent_nan > 0:
            check.add(i)


    for pclass in np.unique(df['Pclass']):
        for sex in np.unique(df['Sex']):
            for title in check:
                mask = df['Pclass'] == pclass
                mask2 = df['Sex'] == sex
                mask3 = df['Title'] == title
                percent_nan = count_nans(df[mask & mask2 & mask3], ['Age'], verbose=False)[0][1]
                if percent_nan > 0:
                    imputer = Imputer(strategy='mean')
                    df.loc[mask & mask2 & mask3, 'Age'] = imputer.fit_transform(df.loc[mask & mask2 & mask3, 'Age'].values.reshape(-1,1))


def count_nans(df, columns, verbose=True):
    """
    Calculates nan value percentages per column in a pandas DataFrame.

    Parameters:
    ----------
    df : (Pandas DataFrame)
    columns : (list)
        A list of strings of the columns to check for null values. Note
        that even if you are checking only one column, it must be
        contained within a list. (Pass df.columns to check all columns)
    verbose : (bool)
        If True (default), prints column names and NaN percentage

    Returns:
    ----------
    col_nans : (list)
        List containing tuples of column names and percentage NaN for
        that column.
    """
    col_nans = []
    for col in columns:
        percent_nan = pd.isna(df[col]).sum()/len(pd.isna(df[col]))
        col_nans.append((col, percent_nan))
        if verbose:
            print("{} | {:.2f}% NaN".format(col, percent_nan*100))

    return col_nans


def count_nans_array(X):
    nans = []
    for i in range(X.shape[1]):
        print("Column: {} | NaNs: {}".format(i, np.isnan(X[:, i]).sum()))
        nans.append(np.isnan(X[:, i]).sum())

    return np.array(nans)


def score_model(model, x_train, y_train, cv=5, verbose=True):
    '''
    Performs K-Fold cross validation for the inputted model and returns the
    accuracy, precision and recall, each averaged over the number of folds
    specified by cv.

    Parameters:
    ----------
    model : (sklearn model object)
        An instantiated model object that implements the .fit() method
    x_train : (2D array)
        Training data
    y_train : (1D array)
        Training data target variable values
    cv : (int)
        Number of folds for K-Fold cross validation (default=5)
    verbose: (bool)
        If True (default), prints the accuracy, precision and recall

    Returns:
    ----------
    cv_acc : (1D array)
        Cross validated accuracy scores; will have length equal to cv
    cv_prec : (1D array)
        Cross validated precision scores; will have length equal to cv
    cv_rec : (1D array)
        Cross validated recall scores; will have length equal to cv
    '''
    cv_acc = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    cv_prec = cross_val_score(model, x_train, y_train, scoring='precision', cv=cv, n_jobs=-1)

    cv_rec = cross_val_score(model, x_train, y_train, scoring='recall', cv=cv, n_jobs=-1)

    if verbose:
        print("{} | {}-Fold Accuracy: {:.4f}".format(model.__class__.__name__, cv, np.mean(cv_acc)))
        print("{} | {}-Fold Precision: {:.4f}".format(model.__class__.__name__, cv, np.mean(cv_prec)))
        print("{} | {}-Fold Recall: {:.4f}".format(model.__class__.__name__, cv, np.mean(cv_rec)))

    return cv_acc, cv_prec, cv_rec


if __name__ == "__main__":

    # data prep
    df = pd.read_csv("../titanic_train.csv")
    format_data(df)
    impute_age(df)
    df.drop(np.argwhere(pd.isnull(df['Embarked'].values)).ravel(), inplace=True)
    X = df.drop(['PassengerId','Survived', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=False)
    X = pd.get_dummies(X).values
    y = df['Survived'].values

    # benchmark modeling
    np.random.seed(5)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    log_mod = LogisticRegression(penalty='l1')
    log_acc, log_prec, log_rec = score_model(log_mod, x_train, y_train, 8)
