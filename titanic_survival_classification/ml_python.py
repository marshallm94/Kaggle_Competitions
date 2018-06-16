import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
plt.style.use('ggplot')


def clean_data(df):
    df['Pclass'] = df['Pclass'].astype(object)
    df.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
    df = pd.get_dummies(df)

    X = df.drop('Survived', axis=1, inplace=False).values
    y = df['Survived'].values

    impute = Imputer()
    X = impute.fit_transform(X)

    return X, y


def count_nans_df(df):
    nans = []
    for i in df.select_dtypes(exclude=['object']).columns:
        print(i, np.isnan(df[i].values).sum())
        nans.append(np.isnan(df[i].values).sum())

    return list(zip(list(df.columns), nans))


def count_nans_array(X):
    nans = []
    for i in range(X.shape[1]):
        print("Column: {} | NaNs: {}".format(i, np.isnan(X[:, i]).sum()))
        nans.append(np.isnan(X[:, i]).sum())

    return np.array(nans)


def score_model(model, x_train, y_train, cv=5):
    acc = np.mean(cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1))

    prec = np.mean(cross_val_score(model, x_train, y_train, scoring='precision', cv=cv, n_jobs=-1))

    rec = np.mean(cross_val_score(model, x_train, y_train, scoring='recall', cv=cv, n_jobs=-1))

    print("{} | {}-Fold Accuracy: {}".format(model.__class__.__name__, cv, acc))
    print("{} | {}-Fold Precision: {}".format(model.__class__.__name__, cv, prec))
    print("{} | {}-Fold Recall: {}".format(model.__class__.__name__, cv, rec))

    return acc, prec, rec


if __name__ == "__main__":

    # data cleaning
    df = pd.read_csv("/Users/marsh/data_science_projects/Kaggle_Competitions/titanic_survival_classification/titanic_train.csv")
    X, y = clean_data(df)

    standardizer = StandardScaler()
    standardizer.fit(X)
    X = standardizer.transform(X)

    # modeling
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    log_mod = LogisticRegression()
    rf = RandomForestClassifier()
    gradient_booster = GradientBoostingClassifier()
    svc = SVC()
    dt = DecisionTreeClassifier()
    bag = BaggingClassifier()

    models = [log_mod, rf, gradient_booster, svc, dt, bag]
    for model in models:
        score_model(model, x_train, y_train)

    # coarse grid searching
    log_mod_grid = {"penalty": ['l1','l2'],
                    "C": [0.001, 0.01, 0.1, 1, 10],
                    "class_weight": ['balanced', None]
    }
    rf_grid = {"n_estimators": list(np.arange(100, 700, 100)),
               "criterion": ['gini','entropy'],
               "max_features": ['auto','log2', None]
    }
    gradient_booster_grid = {"learning_rate": [0.001, 0.01, 0.1, 1, 10],
                             "n_estimators": list(np.arange(100, 700, 100)),
                             "max_depth": [1,2,3],
    }
    svc_grid = {"C": [1, 3, 5, 10],
                "kernel": ['linear','poly','rbf'],
                "degree": [2,3,4]
    }
    dt_grid = {"criterion": ['gini','entropy'],
               "class_weight": ["balanced", None]
    }
    bag_grid = {"n_estimators": list(np.arange(100, 700, 100)),
                "max_features": [0.25, 0.5, 0.75, 1.0]
    }

    grids = [log_mod_grid, rf_grid, gradient_booster_grid, svc_grid, dt_grid, bag_grid]

    model_grids = zip(models, grids)
    for model, grid in model_grids:
        g = GridSearchCV(model, n_jobs=-1, scoring='accuracy', cv=5, param_grid=grid, verbose=True)
        g.fit(x_train, y_train)
        score_model(g.best_estimator_, x_train, y_train)
