from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    mean_squared_error, pairwise_distances, pairwise_distances_argmin,
    confusion_matrix, classification_report
                             )
from sklearn.calibration import calibration_curve

data = pd.read_csv('data.csv')

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'eta': [0.2, 0.1, 0.05],
    'n_estimators': np.arange(50, 301, 50)
}


def xgb_classifier_tuning(X_train, y_train, param_grid, multi_class=False):
    """
    A function to perform hyperparameter tuning on an XGBoost classifier using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training data features.
    - y_train (pd.Series): Training data target.
    - param_grid (dict): Dictionary with hyperparameters to test.
    - multi_class (bool): if True then number of classes >2

    Returns:
    - best_params (dict): Dictionary with the best hyperparameters found.
    - best_score (float): Best score obtained.
    """
    # calculate weights to address imbalanced data
    if multi_class:
        weight_dict = dict()
        for k, v in y_train.value_counts().to_dict().items():
            weight_dict[k] = v / y_train.shape[0]
        weights = [weight_dict[i] for i in y_train.values]
        xgb_model = xgb.XGBClassifier(weight=weights,
                                      objective='multi:softmax',
                                      verbosity=0)

        scoring = 'f1_weighted'

    else:
        count_negatives = len(y_train) - sum(y_train)
        count_positives = sum(y_train)
        scale_pos_weight = count_negatives / count_positives

        # Instantiate the XGBoost classifier
        xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                      objective='binary:logistic',
                                      eval_metric='auc',
                                      verbosity=0)
        scoring = 'roc_auc'

    # Instantiate the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring=scoring)

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # print("Best hyperparameters found:\n", best_params)
    # print("Best score:", best_score)

    best_model = grid_search.best_estimator_

    # Return the best hyperparameters and best score
    return best_params, best_score, best_model


def bootstrap(data, num_iter):
    print("Started at: ", datetime.now().strftime("%H:%M:%S"))
    ATE = {
        's': [],
        't': [],
        'ipw': []
    }

    for i in range(num_iter):
        sample = data.sample(data.shape[0], replace=True)
        X_sample = sample.drop(['T', 'HUMRAT_TEUNA'], axis=1)
        T_sample = sample['T']
        Y_sample = sample['HUMRAT_TEUNA']

        # S-learner. X includes T in this model
        X_s = sample.copy().drop(['HUMRAT_TEUNA'], axis=1)
        Y_s = sample['HUMRAT_TEUNA']
        X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, Y_s,
                                                                    random_state=27)
        best_params_s, best_score_s, best_model_s = xgb_classifier_tuning(
            X_s_train, y_s_train, param_grid, multi_class=True)
        X_t0 = X_s.copy()
        X_t0['T'] = 0

        X_t1 = X_s.copy()
        X_t1['T'] = 1

        ATE['s'].append(
            (best_model_s.predict(X_t1) - best_model_s.predict(X_t0)).mean())

        # T-Learner
        X_t_0 = sample[sample['T'] == 0].copy().drop(['T', 'HUMRAT_TEUNA'],
                                                     axis=1)
        X_t_1 = sample[sample['T'] == 1].copy().drop(['T', 'HUMRAT_TEUNA'],
                                                     axis=1)

        Y_t_0 = sample[sample['T'] == 0]['HUMRAT_TEUNA'].copy()
        Y_t_1 = sample[sample['T'] == 1]['HUMRAT_TEUNA'].copy()

        Xt0_train, Xt0_test, yt0_train, yt0_test = train_test_split(X_t_0,
                                                                    Y_t_0,
                                                                    random_state=27)
        Xt1_train, Xt1_test, yt1_train, yt1_test = train_test_split(X_t_1,
                                                                    Y_t_1,
                                                                    random_state=27)

        best_params_0, best_score_0, best_model_0 = xgb_classifier_tuning(
            Xt0_train, yt0_train, param_grid, multi_class=True)
        best_params_1, best_score_1, best_model_1 = xgb_classifier_tuning(
            Xt1_train, yt1_train, param_grid, multi_class=True)

        ATE['t'].append((best_model_1.predict(X_sample) - best_model_0.predict(
            X_sample)).mean())

        # IPW
        X_ipw_train, X_ipw_test, T_ipw_train, T_ipw_test = train_test_split(
            X_sample, T_sample, random_state=27)
        best_params_ps, best_score_ps, best_model_ps = xgb_classifier_tuning(
            X_ipw_train, T_ipw_train, param_grid, multi_class=False)

        e_x = best_model_ps.predict_proba(X_sample)[:, 1]

        ATE['ipw'].append((T_sample * Y_sample / e_x).mean() - (
                    (1 - T_sample) * Y_sample / (1 - e_x)).mean())

        if (i + 1) % 10 == 0:
            pd.DataFrame(data=ATE).to_csv('bootstrap_ate.csv')

        print(f"Iteration {i + 1} ended at ",
              datetime.now().strftime("%H:%M:%S"))

    return ATE


ate = bootstrap(data, 1000)
pd.DataFrame(data=ate).to_csv('bootstrap_ate_final.csv')
