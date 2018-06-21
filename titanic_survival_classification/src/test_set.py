import pickle
from benchmark_model import *


if __name__ == "__main__":

    # data prep
    training_data = pd.read_csv('../training_data.csv')
    df = pd.read_csv("../titanic_test.csv")
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

    print(X.shape)

    file_dict = {"Logistic Regression": 'logistic_regression_model.pickle', "Decision Forest": 'decision_forest_model.pickle'}

    for name, filepath in file_dict.items():
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            y_hat = model.predict(X)
            # validation_set_acc = accuracy_score(y, y_hat)
            # print("\n{} Test Set Accuracy: {:.2f}%".format(name, validation_set_acc*100))
