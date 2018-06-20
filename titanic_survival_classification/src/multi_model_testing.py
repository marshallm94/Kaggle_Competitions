from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from benchmark_model import *

def distribution_plot(df, column_name, xlab, ylab, title, filename=False, plot_type="box", order=None):
    """
    Create various plot types leverage matplotlib.
    Inputs:
        df: (Pandas DataFrame)
        column_name: (str) - A column in df that you want to have on the x-axis
        target_column: (str) - A column in df that you want to have on the y_axis
        xlab, ylab, title: (all str) - Strings for the x label, y label and title of the plot, respectively.
        filename: (str) - the relative path to which you would like to save the image
        plot_type: (str) - "box", "violin" or "bar"
        order: (None (default) or list) - the ordering of the variable on the x-axis
    Output:
        None (displays figure and saves image)
    """
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    if plot_type == "box":
        ax = sns.boxplot(df[column_name], order=order)
    elif plot_type == "violin":
        ax = sns.violinplot(df[column_name])
    elif plot_type == "bar":
        ax = sns.barplot(df[column_name], palette="Greens_d", order=order)
    ax.set_xlabel(xlab, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=14)
    plt.xticks(rotation=75)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    if filename:
        plt.savefig(filename)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":

    # data prep
    df = pd.read_csv("../titanic_train.csv")
    format_data(df)
    impute_age(df)
    df.drop(np.argwhere(pd.isnull(df['Embarked'].values)).ravel(), inplace=True)
    X = df.drop(['PassengerId','Survived', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=False)
    X = pd.get_dummies(X).values
    y = df['Survived'].values
    np.random.seed(5)
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    rf = RandomForestClassifier()
    gradient_booster = GradientBoostingClassifier()
    svc = SVC()
    dt = DecisionTreeClassifier()
    bag = BaggingClassifier()

    model_dict = {"Random Forest": rf, "Gradient Boost": gradient_booster, "SVM": svc, "Decision Tree": dt, "Bagged Trees": bag}
    acc_dict = {}
    prec_dict = {}
    rec_dict = {}
    for name, model in model_dict.items():
        acc, prec, rec = score_model(model, x_train, y_train, 8)
        acc_dict[name] = acc
        prec_dict[name] = prec
        rec_dict[name] = rec

    acc_df = pd.DataFrame(acc_dict)
    prec_df = pd.DataFrame(prec_dict)
    rec_df = pd.DataFrame(rec_dict)

    distribution_plot(acc_df, "Bagged Trees", 'xtest','ytest','title_test')

    # coarse grid searching
    # log_mod_grid = {"penalty": ['l1','l2'],
    #                 "C": [0.001, 0.01, 0.1, 1, 10],
    #                 "class_weight": ['balanced', None]
    # }
    # rf_grid = {"n_estimators": list(np.arange(100, 700, 100)),
    #            "criterion": ['gini','entropy'],
    #            "max_features": ['auto','log2', None]
    # }
    # gradient_booster_grid = {"learning_rate": [0.001, 0.01, 0.1, 1, 10],
    #                          "n_estimators": list(np.arange(100, 700, 100)),
    #                          "max_depth": [1,2,3],
    # }
    # svc_grid = {"C": [1, 3, 5, 10],
    #             "kernel": ['linear','poly','rbf', 'sigmoid','precomputed'],
    #             "degree": [2,3,4]
    # }
    # dt_grid = {"criterion": ['gini','entropy'],
    #            "class_weight": ["balanced", None]
    # }
    # bag_grid = {"n_estimators": list(np.arange(100, 700, 100)),
    #             "max_features": [0.25, 0.5, 0.75, 1.0]
    # }
    #
    # grids = [rf_grid, gradient_booster_grid, svc_grid, dt_grid, bag_grid]
    #
    # model_grids = zip(models, grids)
    # for model, grid in model_grids:
    #     g = GridSearchCV(model, n_jobs=-1, scoring='accuracy', cv=8, param_grid=grid, verbose=True)
    #     g.fit(x_train, y_train)
    #     score_model(g.best_estimator_, x_train, y_train)
