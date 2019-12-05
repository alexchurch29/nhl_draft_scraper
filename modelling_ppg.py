import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


def main():
    train_models()
    return


def pre_processing(adj=0, predictor=1, test_size=0.3, random_state=0):
    '''
    preprocess data that will be used to train and test our models
    :param adj: 1 if adjusted stats are to be used, 0 for standard stats
    :param awards: 1 if predictor is player accoloades at nhl level vs p_gp rates
    :param test_size: % of dataframe to be used in train set
    :param random_state: random state for initialization
    :param one_hot: applies one hot encoding to categorical variables
    :param feature_scale: applies feature scaling to input variables
    :return: original dataset, along with train/test sets for both features and predictors
    '''
    conn = sqlite3.connect('nhl_draft.db')

    df = pd.read_sql_query('''
                    select t1.player_id as player_id, height, weight, age, pos, country, league_id17, league_id18, 
                    g_gp17, a_gp17, g_gp18, a_gp18, 
                    adj_g_a17, adj_g_a18, adj_a_a17, adj_a_a18, wjc_18, wjc_20, player_classifier,
                    case when t2.all_star > 0 then 1 else 0 end as all_star, case when t5.round = 1 then 1 else 0 end as first_round
                    from
                    (select player_id, age, height, weight, birth_year, pos, country,
                    max(Case when age2 = 17 then normalized_g_gp end) as adj_g_a17,
                    max(Case when age2 = 18 then normalized_g_gp end) as adj_g_a18,
                    max(Case when age2 = 17 then normalized_a_gp end) as adj_a_a17,
                    max(Case when age2 = 18 then normalized_a_gp end) as adj_a_a18,
                    max(Case when age2 = 17 then gp end) as adj_gp_a17,
                    max(Case when age2 = 18 then gp end) as adj_gp_a18
                    from 
                        (select t1.player_id as player_id, age2, age, height, weight, pos, country,
                        cast(substr(dob,0, 5) as int) as birth_year,
                        sum(gp) as gp, sum(adj_g_gp * le * gp) / sum(gp) as normalized_g_gp, 
                        sum(adj_a_gp * le * gp) / sum(gp) as normalized_a_gp, sum(gp) as gp
                        from
                        (select *, t1.g_gp * era_adj_p_gp as adj_g_gp, t1.a_gp * era_adj_p_gp as adj_a_gp, t2.le 
                        from skater_stats_season t1
                        inner join league_equivalencies t2
                        on t1.league_name = t2.league_name
                        inner join era_adjustments t3
                        on t1.league_name = t3.league_name and t1.season = t3.season) t1
                        inner join bios t2 
                        on t1.player_id = t2.player_id
                        where pos2="F" and cast(substr(dob,0, 5) as int)<=1994
                        group by t1.player_id, age2) t1
                    group by player_id) t1
                    left join awards t2
                    on t1.player_id=t2.player_id
                    left join(select player_id,
                    max(Case when age = 17 then league_id end) as league_id17,
                    max(Case when age = 17 then gp end) as gp17,
                    max(Case when age = 17 then g_gp end) as g_gp17,
                    max(Case when age = 17 then a_gp end) as a_gp17,
                    max(Case when age = 18 then league_id end) as league_id18,
                    max(Case when age = 18 then gp end) as gp18,
                    max(Case when age = 18 then g_gp end) as g_gp18,
                    max(Case when age = 18 then a_gp end) as a_gp18
                    from(select player_id, age2 as age, max(gp) as gp, league_id, g_gp, a_gp
                    from skater_stats_season
                    group by player_id, age)
                    group by player_id) t3
                    on t1.player_id=t3.player_id 
                    left join (
                    select player_id, gp, p_gp, 
                    case when gp /* < 1 */ <= 164 or p_gp < 0.33 then 0  
                    else 1 end as player_classifier
                    from skater_stats_career
                    where league_name = "NHL") t4
                    on t1.player_id=t4.player_id
                    left join draft t5 
                    on t1.player_id = t5.player_id
                    where adj_g_a17 notnull and adj_g_a18 and adj_a_a17 notnull and adj_a_a18 notnull and height notnull 
                    and age notnull and weight notnull and pos notnull and country notnull and gp17>=20 
                    and gp18>=20 and league_id17 notnull and league_id18 notnull
                    ''', conn)

    df[['WJC_18', 'WJC_20', 'player_classifier', 'all_star', 'first_round']] = \
        df[['WJC_18', 'WJC_20', 'player_classifier', 'all_star', 'first_round']].fillna(0)

    df = one_hot_encoding(df)
    if adj:
        X = df[[i for i in df.columns if i not in ['first_round', 'all_star', 'player_classifier', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']]]
    else:
        X = df[[i for i in df.columns if i not in ['first_round', 'all_star', 'player_classifier', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18']]]
    if predictor == 1:
        y = df[['player_classifier']]
    elif predictor == 2:
        y = df[['all_star']]
    elif predictor == 3:
        y = df[['first_round']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    X_train, X_test = feature_scaling(X_train, X_test, adj)

    return df, X_train, X_test, y_train, y_test


def one_hot_encoding(df):
    '''
    :param df: dataframe to tranform
    :return: transformed df
    '''
    # league_id17, league_id18, pos, country

    df = pd.concat([df, pd.get_dummies(df['league_id17'], prefix='league_id17')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['league_id18'], prefix='league_id18')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['pos'], prefix='pos')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)

    df.drop(['league_id17'], axis=1, inplace=True)
    df.drop(['league_id18'], axis=1, inplace=True)
    df.drop(['pos'], axis=1, inplace=True)
    df.drop(['country'], axis=1, inplace=True)

    return df


def feature_scaling(train, test, adj):
    '''
    :param df: dataframe to tranform
    :return: transformed df
    '''
    if adj:
        train_data_to_standardize = train[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'WJC_18', 'WJC_20']]
        test_data_to_standardize = test[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'WJC_18', 'WJC_20']]
        scaler = StandardScaler().fit(train_data_to_standardize)
        train_standardized_columns = scaler.transform(train_data_to_standardize)
        test_standardized_columns = scaler.transform(test_data_to_standardize)
        train[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'WJC_18', 'WJC_20']] = train_standardized_columns
        test[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'WJC_18', 'WJC_20']] = test_standardized_columns
    else:
        train_data_to_standardize = train[['height', 'weight', 'age', 'WJC_18', 'WJC_20', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']]
        test_data_to_standardize = test[['height', 'weight', 'age', 'WJC_18', 'WJC_20', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']]
        scaler = StandardScaler().fit(train_data_to_standardize)
        train_standardized_columns = scaler.transform(train_data_to_standardize)
        test_standardized_columns = scaler.transform(test_data_to_standardize)
        train[['height', 'weight', 'age', 'WJC_18', 'WJC_20', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']] = train_standardized_columns
        test[['height', 'weight', 'age', 'WJC_18', 'WJC_20', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']] = test_standardized_columns

    return train, test


def select_features(X_train, y_train, X_test, algorithm, threshold=0.05):
    '''
    Select optimal subset of features fro Train set
    :param X_train: training set
    :param y_train: training predictors
    :param X_test: test set
    :param threshold: threshold for feature selection
    :param algorithm: feature selection algorithm
    :return: optimal subset of training set
    '''
    clf = algorithm
    sfm = SelectFromModel(clf, threshold=threshold)
    sfm.fit(X_train, y_train.values.ravel())
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    return X_train, X_test


def save_model(clf, filename):
    '''
    save a trained model to be used later on
    :param clf: classifier to save
    :return: None
    '''
    dump(clf, filename)

    return


def train_models(cv=3):
    '''
    train models
    :return:
    '''

    df = pre_processing(adj=1, predictor=1)
    X_train_original = df[1]
    X_test_original = df[2]
    y_train = df[3]
    y_test = df[4]

    X_train = X_train_original[[i for i in X_train_original.columns if i not in ['player_id']]]
    X_test = X_test_original[[i for i in X_test_original.columns if i not in ['player_id']]]

    # X_train, X_test = select_features(X_train, y_train, X_test, LassoCV())

    smt = SMOTE()
    nr = NearMiss()
    smt_X_train, smt_y_train = smt.fit_sample(X_train, y_train)
    # nr_X_train, nr_y_train = nr.fit_sample(X_train, y_train)

    clf = GradientBoostingClassifier()
    clf.fit(smt_X_train, smt_y_train)
    smt_y_train_nbc_pred = cross_val_predict(clf, smt_X_train, smt_y_train, cv=cv)
    y_pred = clf.predict(X_test)
    print("SMOTE")
    print("f1: {}".format(f1_score(smt_y_train, smt_y_train_nbc_pred)))
    print(confusion_matrix(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # clf = GradientBoostingClassifier()
    # clf.fit(nr_X_train, nr_y_train)
    # nr_y_train_nbc_pred = cross_val_predict(clf, nr_X_train, nr_y_train, cv=cv)
    # y_pred = clf.predict(X_test)
    # print("Near Miss")
    # print("f1: {}".format(f1_score(nr_y_train, nr_y_train_nbc_pred)))
    # print(confusion_matrix(y_test, y_pred))
    # print("Recall:", recall_score(y_test, y_pred))
    # print("Accuracy:", accuracy_score(y_test, y_pred))

    # save_model(clf, 'clf_allstar_player.sav')

    test = X_test_original.loc[:, [i for i in X_test_original.columns if i in ['player_id']]]
    test['player_class'] = y_test
    test['prediction'] = y_pred
    test['probability'] = [i[1] for i in clf.predict_proba(X_test)]
    test = test.sort_values(by=['prediction', 'probability'], ascending=False)
    test.to_sql('test', sqlite3.connect('nhl_draft.db'), if_exists='replace', index=False)

    '''
    vclf = VotingClassifier(
        estimators=[('gbc', GradientBoostingClassifier()), ('mlpc', MLPClassifier()), ('rfc', RandomForestClassifier()),
                    ('knn', KNeighborsClassifier())],
        voting='soft'
    )
    vclf.fit(X_train, y_train.values.ravel())
    y_train_vclf_pred = cross_val_predict(vclf, X_train, y_train.values.ravel(), cv=cv)
    print("vclf: {}".format(f1_score(y_train.values.ravel(), y_train_vclf_pred, average="macro")))

    gbc_clf = GradientBoostingClassifier()
    gbc_clf.fit(X_train, y_train.values.ravel())
    y_train_gbc_pred = cross_val_predict(gbc_clf, X_train, y_train.values.ravel(), cv=cv)
    print("gbc: {}".format(f1_score(y_train.values.ravel(), y_train_gbc_pred, average="macro")))

    mlpc_clf = MLPClassifier()
    mlpc_clf.fit(X_train, y_train.values.ravel())
    y_train_mlpc_pred = cross_val_predict(mlpc_clf, X_train, y_train.values.ravel(), cv=cv)
    print("mlpc: {}".format(f1_score(y_train.values.ravel(), y_train_mlpc_pred, average="macro")))

    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(X_train, y_train.values.ravel())
    y_train_rfc_pred = cross_val_predict(rfc_clf, X_train, y_train.values.ravel(), cv=cv)
    print("rfc: {}".format(f1_score(y_train.values.ravel(), y_train_rfc_pred, average="macro")))

    nbc_clf = GaussianNB()
    nbc_clf.fit(X_train, y_train.values.ravel())
    y_train_nbc_pred = cross_val_predict(nbc_clf, X_train, y_train.values.ravel(), cv=5)
    print("nbc: {}".format(f1_score(y_train.values.ravel(), y_train_nbc_pred, average="macro")))

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train.values.ravel())
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train.values.ravel(), cv=cv)
    print("knn: {}".format(f1_score(y_train.values.ravel(), y_train_knn_pred, average="macro")))

    dtc_clf = DecisionTreeClassifier()
    dtc_clf.fit(X_train, y_train.values.ravel())
    y_train_dtc_pred = cross_val_predict(dtc_clf, X_train, y_train.values.ravel(), cv=cv)
    print("dtc: {}".format(f1_score(y_train.values.ravel(), y_train_dtc_pred, average="macro")))

    svc_clf = SVC()
    svc_clf.fit(X_train, y_train.values.ravel())
    y_train_svc_pred = cross_val_predict(svc_clf, X_train, y_train.values.ravel(), cv=cv)
    print("svc: {}".format(f1_score(y_train.values.ravel(), y_train_svc_pred, average="macro")))
    '''

    return


if __name__ == '__main__':
    main()
