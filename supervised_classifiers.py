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
    # pre_processing()
    train_models(adj=1, clusters=0, classifier=2, imb=1)
    return


def pre_processing(test_size=0.3, random_state=0):
    '''
    preprocess data that will be used to train and test our models
    :param test_size: % of dataframe to be used in train set
    :param random_state: random state for initialization
    '''
    conn = sqlite3.connect('nhl_draft.db')

    df = pd.read_sql_query('''
                    select t1.player_id as player_id, height, weight, age, pos, country, league_id17, league_id18, 
                    g_gp17, a_gp17, g_gp18, a_gp18, 
                    adj_g_a17, adj_g_a18, adj_a_a17, adj_a_a18, wjc_18, wjc_20, clusters50, 
                    clusters100, clusters200, classifier1, classifier2, 
                    case when t2.all_star > 0 then 1 else 0 end as classifier3
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
                    case when gp <= 164 or p_gp < 0.33 then 0  
                    else 1 end as classifier2
                    from skater_stats_career
                    where league_name = "NHL") t4
                    on t1.player_id=t4.player_id
                    left join draft t5 
                    on t1.player_id = t5.player_id
                    left join forward_clusters t6
                    on t1.player_id = t6.player_id
                    left join (
                    select player_id, gp, p_gp, 
                    case when gp < 1 then 0  
                    else 1 end as classifier1
                    from skater_stats_career
                    where league_name = "NHL") t7
                    on t1.player_id=t7.player_id
                    where adj_g_a17 notnull and adj_g_a18 and adj_a_a17 notnull and adj_a_a18 notnull and height notnull 
                    and age notnull and weight notnull and pos notnull and country notnull and gp17>=20 
                    and gp18>=20 and league_id17 notnull and league_id18 notnull
                    ''', conn)

    df[['WJC_18', 'WJC_20', 'classifier1', 'classifier2', 'classifier3', 'clusters50', 'clusters100', 'clusters200']] = \
        df[['WJC_18', 'WJC_20', 'classifier1', 'classifier2', 'classifier3', 'clusters50', 'clusters100', 'clusters200']].fillna(0)

    X = df[[i for i in df.columns if i not in ['classifier1', 'classifier2', 'classifier3']]]
    y = df[['classifier1']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_test = feature_scaling(X_train, X_test)

    X_train.to_sql('X_train1', conn, if_exists='replace', index=False)
    X_test.to_sql('X_test1', conn, if_exists='replace', index=False)
    y_train.to_sql('y_train1', conn, if_exists='replace', index=False)
    y_test.to_sql('y_test1', conn, if_exists='replace', index=False)

    X = df[[i for i in df.columns if i not in ['classifier1', 'classifier2', 'classifier3']]]
    y = df[['classifier2']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    X_train, X_test = feature_scaling(X_train, X_test)

    X_train.to_sql('X_train2', conn, if_exists='replace', index=False)
    X_test.to_sql('X_test2', conn, if_exists='replace', index=False)
    y_train.to_sql('y_train2', conn, if_exists='replace', index=False)
    y_test.to_sql('y_test2', conn, if_exists='replace', index=False)

    X = df[[i for i in df.columns if i not in ['classifier1', 'classifier2', 'classifier3']]]
    y = df[['classifier3']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    X_train, X_test = feature_scaling(X_train, X_test)

    X_train.to_sql('X_train3', conn, if_exists='replace', index=False)
    X_test.to_sql('X_test3', conn, if_exists='replace', index=False)
    y_train.to_sql('y_train3', conn, if_exists='replace', index=False)
    y_test.to_sql('y_test3', conn, if_exists='replace', index=False)

    return


def one_hot_encoding(X_train, X_test, cluster):
    '''
    :param X_train: training set to tranform
    :param X_test: test set to tranform
    :param cluster: specify which level of clustering is being used (0 if none)
    :return: transformed df
    '''
    if cluster == 1:
        df = pd.concat([X_train[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters50']],
             X_test[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters50']]])
        df = pd.concat([df, pd.get_dummies(df['league_id17'], prefix='league_id17')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['league_id18'], prefix='league_id18')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['pos'], prefix='pos')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['clusters50'], prefix='country')], axis=1)
        df.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters50'], axis=1, inplace=True)
        X_train.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters50'], axis=1, inplace=True)
        X_test.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters50'], axis=1, inplace=True)

    elif cluster == 2:
        df = pd.concat([X_train[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters100']],
                        X_test[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters100']]])
        df = pd.concat([df, pd.get_dummies(df['league_id17'], prefix='league_id17')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['league_id18'], prefix='league_id18')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['pos'], prefix='pos')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['clusters100'], prefix='country')], axis=1)
        df.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters100'], axis=1, inplace=True)
        X_train.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters100'], axis=1, inplace=True)
        X_test.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters100'], axis=1, inplace=True)

    elif cluster == 3:
        df = pd.concat([X_train[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters200']],
                        X_test[['player_id', 'league_id17', 'league_id18', 'pos', 'country', 'clusters200']]])
        df = pd.concat([df, pd.get_dummies(df['league_id17'], prefix='league_id17')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['league_id18'], prefix='league_id18')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['pos'], prefix='pos')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['clusters200'], prefix='country')], axis=1)
        df.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters200'], axis=1, inplace=True)
        X_train.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters200'], axis=1, inplace=True)
        X_test.drop(['league_id17', 'league_id18', 'pos', 'country', 'clusters200'], axis=1, inplace=True)
    
    else:
        df = pd.concat([X_train[['player_id', 'league_id17', 'league_id18', 'pos', 'country']],
             X_test[['player_id', 'league_id17', 'league_id18', 'pos', 'country']]])
        df = pd.concat([df, pd.get_dummies(df['league_id17'], prefix='league_id17')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['league_id18'], prefix='league_id18')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['pos'], prefix='pos')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
        df.drop(['league_id17', 'league_id18', 'pos', 'country'], axis=1, inplace=True)
        X_train.drop(['league_id17', 'league_id18', 'pos', 'country'], axis=1, inplace=True)
        X_test.drop(['league_id17', 'league_id18', 'pos', 'country'], axis=1, inplace=True)

    X_train.merge(df, on='player_id', how='left')
    X_test.merge(df, on='player_id', how='left')

    return X_train, X_test


def feature_scaling(X_train, X_test):
    '''
    normalize ordinal features
    :param X_train: training set to tranform
    :param X_test: test set to tranform
    :return: transformed train/test sets
    '''
    X_train_data_to_standardize = X_train[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18', 'WJC_18', 'WJC_20']]
    X_test_data_to_standardize = X_test[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18', 'WJC_18', 'WJC_20']]
    scaler = StandardScaler().fit(X_train_data_to_standardize)
    X_train_standardized_columns = scaler.transform(X_train_data_to_standardize)
    X_test_standardized_columns = scaler.transform(X_test_data_to_standardize)
    X_train[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18', 'WJC_18', 'WJC_20']] = X_train_standardized_columns
    X_test[['height', 'weight', 'age', 'adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18', 'g_gp17', 'a_gp17', 'g_gp18', 'a_gp18', 'WJC_18', 'WJC_20']] = X_test_standardized_columns

    return X_train, X_test


def select_features(X_train, y_train, X_test, algorithm, threshold=0.01):
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


def train_models(adj=0, clusters=0, classifier=1, imb=0):
    '''
    train models
    :return:
    '''
    import warnings
    warnings.simplefilter(action='ignore', category=Warning)

    conn = sqlite3.connect('nhl_draft.db')

    X_train_original = pd.read_sql_query('''SELECT * FROM X_TRAIN{}'''.format(str(classifier)), conn)
    X_test_original = pd.read_sql_query('''SELECT * FROM X_TEST{}'''.format(str(classifier)), conn)
    y_train = pd.read_sql_query('''SELECT * FROM Y_TRAIN{}'''.format(str(classifier)), conn)
    y_test = pd.read_sql_query('''SELECT * FROM Y_TEST{}'''.format(str(classifier)), conn)

    if adj == 1:
        X_train = X_train_original[[i for i in X_train_original.columns if i not in ['g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']]]
        X_test = X_test_original[[i for i in X_test_original.columns if i not in ['g_gp17', 'a_gp17', 'g_gp18', 'a_gp18']]]
    else:
        X_train = X_train_original[[i for i in X_train_original.columns if i not in ['adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18']]]
        X_test = X_test_original[[i for i in X_test_original.columns if i not in ['adj_g_a17', 'adj_g_a18', 'adj_a_a17', 'adj_a_a18']]]
        
    if clusters == 1:
        X_train = X_train[[i for i in X_train.columns if i not in ['clusters100', 'clusters200']]]
        X_test = X_test[[i for i in X_test.columns if i not in ['clusters100', 'clusters200']]]
    elif clusters == 2:
        X_train = X_train[[i for i in X_train.columns if i not in ['clusters50', 'clusters200']]]
        X_test = X_test[[i for i in X_test.columns if i not in ['clusters50', 'clusters200']]]
    elif clusters == 3:
        X_train = X_train[[i for i in X_train.columns if i not in ['clusters50', 'clusters100']]]
        X_test = X_test[[i for i in X_test.columns if i not in ['clusters50', 'clusters100']]]
    else:
        X_train = X_train[[i for i in X_train.columns if i not in ['clusters50', 'clusters100', 'clusters200']]]
        X_test = X_test[[i for i in X_test.columns if i not in ['clusters50', 'clusters100', 'clusters200']]]
        
    X_train, X_test = one_hot_encoding(X_train, X_test, clusters)
    X_train = X_train[[i for i in X_train.columns if i not in ['player_id']]]
    X_test = X_test[[i for i in X_test.columns if i not in ['player_id']]]
    X_train, X_test = select_features(X_train, y_train, X_test, GradientBoostingClassifier())
    
    models = [GradientBoostingClassifier(), MLPClassifier(), SVC(), RandomForestClassifier(), GaussianNB(), KNeighborsClassifier()]
    # models = [VotingClassifier(estimators=[('gbc', GradientBoostingClassifier()), ('rnc', RandomForestClassifier())],voting='soft'),
    #           VotingClassifier(estimators=[('gbc', GradientBoostingClassifier()), ('rnc', RandomForestClassifier()),
    #                                        ('mlp', MLPClassifier())], voting='soft'),
    #           VotingClassifier(estimators=[('gbc', GradientBoostingClassifier()), ('rnc', RandomForestClassifier()),
    #                                        ('knn', KNeighborsClassifier())], voting='soft')]
    for model in models:
        if imb == 1:
            smt = SMOTE()
            X_train, y_train = smt.fit_sample(X_train, y_train)
        elif imb == 2:
            nr = NearMiss()
            X_train, y_train = nr.fit_sample(X_train, y_train)
        clf = model
        clf.fit(X_train, y_train)
        y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
        y_pred = clf.predict(X_test)
        print(type(model).__name__)
        print(confusion_matrix(y_test, y_pred))
        print("f1: {}".format(f1_score(y_train, y_train_pred)))
        # print("Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print()

        # save_model(clf, 'clf1_f.sav')

        # test = X_test_original.loc[:, [i for i in X_test_original.columns if i in ['player_id']]]
        # test['player_class'] = y_test
        # test['prediction'] = y_pred
        # test['probability'] = [i[1] for i in clf.predict_proba(X_test)]
        # test = test.sort_values(by=['prediction', 'probability'], ascending=False)
        # test.to_sql('clf1_f_clusters', sqlite3.connect('nhl_draft.db'), if_exists='replace', index=False)

    return


if __name__ == '__main__':
    main()
