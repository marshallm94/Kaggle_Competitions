from benchmark_model import *
from multi_model_testing import *
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import pickle


if __name__ == "__main__":

    # data prep
    df = pd.read_csv("../data/titanic_train.csv")
    format_data(df)
    impute_age(df)
    df.drop(np.argwhere(pd.isnull(df['Embarked'].values)).ravel(), inplace=True)
    X = df.drop(['PassengerId','Survived', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=False)
    X = pd.get_dummies(X).values
    y = df['Survived'].values
    np.random.seed(5)
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Larger Grid Search
    bag = BaggingClassifier
    bag_grid = {"n_estimators": list(np.arange(1, 500, 20)),
                "max_features": list(np.arange(0.1, 1.1, 0.1))
    }
    name = ['Decision Forest']
    model = [bag]
    grid = [bag_grid]
    acc_dict, prec_dict, rec_dict, best_estimator = multi_model_grid_search(name, model, grid, x_train, y_train)
    print(best_estimator)

    # Validation set
    decision_forest = BaggingClassifier(n_estimators=350, max_features=13)
    decision_forest.fit(x_train, y_train)
    y_hat = decision_forest.predict(x_test)
    validation_set_acc = accuracy_score(y_test, y_hat)
    print("\nDecision Forest Validation Set Accuracy: {:.2f}%".format(validation_set_acc*100))

    with open('../models/decision_forest_model.pickle', 'wb') as f:
        pickle.dump(decision_forest, f, protocol=pickle.HIGHEST_PROTOCOL)
