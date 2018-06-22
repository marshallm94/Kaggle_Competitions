import pickle
from benchmark_model import *


if __name__ == "__main__":

    # data prep
    training_data = pd.read_csv('../data/training_data.csv')
    df = pd.read_csv("../data/titanic_test.csv")
    format_data(df)
    impute_age(df)
    df.loc[np.argwhere(pd.isnull(df['Fare'].values)).ravel(), "Fare"] = df["Fare"].mean()
    X = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=False)
    X = pd.get_dummies(X)
    missing_columns = set(training_data.columns) - set(X.columns)
    for col in missing_columns:
        X[col] = 0
    X = X[training_data.columns]
    X.drop('Unnamed: 0', axis=1, inplace=True)

    X = X.values

    with open("../models/decision_forest_model.pickle", 'rb') as f:
        model = pickle.load(f)
        y_hat = model.predict(X)

    submission = pd.DataFrame({'PassengerId': np.arange(892, 1310), "Survived": y_hat})
    submission.to_csv("../data/decision_forest_submission.csv", index=False)

    with open("../models/logistic_regression_model.pickle", 'rb') as f:
        model = pickle.load(f)
        y_hat = model.predict(X)

    submission = pd.DataFrame({'PassengerId': np.arange(892, 1310), "Survived": y_hat})
    submission.to_csv("../data/logistic_regression_submission.csv", index=False)
