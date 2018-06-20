from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from benchmark_model import *

def box_plot(dictionary, box_line_color, box_fill_color, xlab, ylab, title, horizontal_line_dict=False, filename=False):
    '''
    Creates a boxplot.

    Parameters:
    ----------
    error_dict : (dict)
        A dictionary where the keys are the labels to go on the x axis and
        the values are 1D arrays
    box_line_color : (str)
        The color for the borders of the boxes
    box_fill_color : (str)
        The color for the boxes to be filled with
    xlab : (str)
        X axis label
    ylab : (str)
        Y axis label
    title : (str)
        Title for the plot
    horizontal_line_dict : (bool/dict)
        If False (default), a horizontal line will not be added to the
        plot. If you would like to add a horizontal line to the plot, pass
        a dictionary which has keys "Y Value", "Color", "Label" whose
        values are the location for the label on the same scale as the
        y axis of the plot, the color of the box surrounding the label, and
        the text for the label, respectively.

        Example: horizontal_line_dict = {"Y Value": 0.83, "Color": "red", "Label": "Benchmark"}

    filename : (bool/str)
        If False (default), show the plot. To save the plot, pass the
        absolute or relative filepath to which the plot should be saved.

    Returns:
    ----------
    None
    '''
    fig, ax = plt.subplots(figsize=(15,8))
    bp = ax.boxplot(dictionary.values(), labels=dictionary.keys(), patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=box_line_color)

    for patch in bp['boxes']:
        patch.set(facecolor=box_fill_color)

    if horizontal_line_dict:
        ax.axhline(horizontal_line_dict['Y Value'], linewidth=1, color=horizontal_line_dict["Color"])
        ax.text(1.01, horizontal_line_dict['Y Value'], horizontal_line_dict['Label'], va='center', ha="left", bbox=dict(facecolor=horizontal_line_dict["Color"]),
        transform=ax.get_yaxis_transform())

    ax.set_xlabel(xlab, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=14)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    plt.subplots_adjust(top=0.9)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def multi_model_grid_search(model_names, model_objects, model_grids, x_train, y_train, grid_cv=8, score_model_cv=8):
    '''
    Performs a grid search for mutliple models and then cross validates the
    best estimator of the grid.

    Note that the element indices in each list
    (model_names, model_objects, model_grids) must match; that is to say,
    if the element at model_names[0] is "Linear Regression", then the
    element at model_objects[0] should be an instantiated LinearRegression
    model object and the element at model_grids[0] should be a dictionary
    with keys being equal to LinearRegression parameter names and values
    being lists of options to pass to those parameters.

    Parameters:
    ----------
    model_names : (list)
        List of strings of the
    model_objects : (list)
        List of model objects to be passed to GridSearchCV
        (must utilize .fit() and .predict())
    model_grids : (list)
        List of dictionaries, where each dictionary is the grid to be
        passed to GridSearchCV
    x_train : (2D array)
        Training data with shape (n x p)
    y_train : (1D array)
        Training data target array with shape (n x 1)
    grid_cv : (int)
        The number of folds to be used in GridSearchCV
    score_model_cv : (int)
        The number of folds to be used in score_model

    Returns:
    ----------
    acc_dict : (dict)
        A dictionary where the keys are the model names and the values are
        the accuracy arrays that are the output of score_model
    pred_dict : (dict)
        A dictionary where the keys are the model names and the values are
        the precision arrays that are the output of score_model
    rec_dict : (dict)
        A dictionary where the keys are the model names and the values are
        the recall arrays that are the output of score_model
    '''
    acc_dict = {}
    prec_dict = {}
    rec_dict = {}
    for name, model, grid in zip(model_names, model_objects, model_grids):
        g = GridSearchCV(model, n_jobs=-1, scoring='accuracy', cv=grid_cv, param_grid=grid, verbose=True)
        g.fit(x_train, y_train)
        acc, prec, rec = score_model(g.best_estimator_, x_train, y_train, score_model_cv)
        acc_dict[name] = acc
        prec_dict[name] = prec
        rec_dict[name] = rec

    return acc_dict, prec_dict, rec_dict

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

    # coarse grid searching
    rf_grid = {"n_estimators": list(np.arange(100, 700, 100)),
               "criterion": ['gini','entropy'],
               "max_features": ['auto','log2', None]
    }
    gradient_booster_grid = {"learning_rate": [0.001, 0.01, 0.1, 1, 10],
                             "n_estimators": list(np.arange(100, 700, 100)),
                             "max_depth": [1,2,3],
    }
    dt_grid = {"criterion": ['gini','entropy'],
               "class_weight": ["balanced", None]
    }
    bag_grid = {"n_estimators": list(np.arange(100, 700, 100)),
                "max_features": [0.25, 0.5, 0.75, 1.0]
    }
    svc_grid = {"C": [1, 3, 5, 10],
                "kernel": ['linear','poly','rbf', 'sigmoid'],
                "degree": [2,3,4]
    }

    names = ['Random Forest','Gradient Boost','Decision Tree','10 Bagged Trees', "SVM"]
    models = [rf, gradient_booster, dt, bag, svc]
    grids = [rf_grid, gradient_booster_grid, dt_grid, bag_grid, svc_grid]

    acc_dict, prec_dict, rec_dict = multi_model_grid_search(names, models, grids, x_train, y_train)


    model_grids = zip(['Random Forest','Gradient Boost','Decision Tree','10 Bagged Trees', "SVM"], [rf, gradient_booster, dt, bag, svc], grids)
    acc_dict = {}
    prec_dict = {}
    rec_dict = {}
    for name, model, grid in model_grids:
        g = GridSearchCV(model, n_jobs=-1, scoring='accuracy', cv=8, param_grid=grid, verbose=True)
        g.fit(x_train, y_train)
        acc, prec, rec = score_model(g.best_estimator_, x_train, y_train, 8)
        acc_dict[name] = acc
        prec_dict[name] = prec
        rec_dict[name] = rec

    acc_horiz_line_dict = {"Y Value": 0.83, "Color": "red", "Label": "Benchmark"}
    box_plot(acc_dict, "darkblue", "skyblue", 'Model','Accuracy','Multi-Model Testing: Accuracy', acc_horiz_line_dict)
