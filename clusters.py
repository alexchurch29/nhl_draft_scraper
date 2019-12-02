import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_samples, silhouette_score
import skfuzzy


def main():
    #parse_clusters()
    #test(2017)
    return


def parse_clusters(p="F"):
    """
    Generate clusters using kmeans algorithm
    :param p: position
    :return:
    """
    conn = sqlite3.connect('nhl_draft.db')
    cur = conn.cursor()

    pre_draft = pd.read_sql_query('''
                    select player_id, first_name, last_name, substr(dob,0, 5) as birth_year, 
                        g_a17, g_a18, a_a17, a_a18 from
                        (select player_id, first_name, last_name, dob, gp, 
                        max(Case when age = 17 then normalized_g_gp end) as g_a17,
                        max(Case when age = 18 then normalized_g_gp end) as g_a18,
                        max(Case when age = 17 then normalized_a_gp end) as a_a17,
                        max(Case when age = 18 then normalized_a_gp end) as a_a18
                        from 
                            (select t1.player_id as player_id, age2 as age, dob, first_name, last_name, 
                            sum(gp) as gp, sum(adj_g_gp * le * gp) / sum(gp) as normalized_g_gp, 
                            sum(adj_a_gp * le * gp) / sum(gp) as normalized_a_gp
                            from
                            (select *, t1.g_gp * era_adj_p_gp as adj_g_gp, t1.a_gp * era_adj_p_gp as adj_a_gp, t2.le 
                            from skater_stats_season t1
                            inner join league_equivalencies t2
                            on t1.league_name = t2.league_name
                            inner join era_adjustments t3
                            on t1.league_name = t3.league_name and t1.season = t3.season) t1
                            inner join bios t2 
                            on t1.player_id = t2.player_id
                            where pos2 = :pos
                            group by t1.player_id, age2) t1
                        group by player_id) 
                    where g_a17 notnull and g_a18 and a_a17 notnull and a_a18 notnull and gp>=20 and dob notnull
                    ''', conn, params={'pos': p})

    # Create the scalar.
    data_to_standardize = pre_draft[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']]
    scaler = StandardScaler().fit(data_to_standardize)

    # Standardize the columns.
    standardized_data = pre_draft.copy()
    standardized_columns = scaler.transform(data_to_standardize)
    standardized_data[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']] = standardized_columns
    df = standardized_data[['g_a17', 'g_a18', 'a_a17', 'a_a18', 'a_a18']]

    from scipy import stats
    import numpy as np
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]

    plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}
    plt.scatter(df['g_a17'], df['g_a18'], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()

    def plot_clusters(data, algorithm, args, kwds):
        labels = algorithm(*args, **kwds).fit_predict(data)
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data['g_a17'], data['g_a18'], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        plt.show()

    plot_clusters(df, KMeans, (), {'n_clusters': 25})

    '''
    # elbow method for computing optimal number of clusters
    Sum_of_squared_distances = []
    K = range(25, 351, 25)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(pre_draft[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']])
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('plots/kmeans_elbow.jpeg')
    '''

    km = KMeans(n_clusters=350)
    km = km.fit(standardized_data[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']])
    pre_draft.loc[:, 'clusters'] = km.labels_

    if p == "F":
        pre_draft.to_sql('forward_clusters', conn, if_exists='replace', index=False)

        drop = cur.executescript('''
                            DROP TABLE IF EXISTS forward_clusters2;''')

        cluster_details = cur.executescript('''create table forward_clusters2 as
            select clusters, count(player_id) as n, sum(gp)/count(player_id) as gp, 
            sum(g)/count(player_id) as g, sum(p)/count(player_id) as p/*, 
            case when sum(g)/sum(gp) isnull then 0 else round(sum(g),2)/round(sum(gp),2) end as g_gp, 
            case when sum(p)/sum(gp) isnull then 0 else round(sum(p),2)/round(sum(gp),2) end as p_gp*/
            from (select t1.*, 
            case when t2.gp notnull then t2.gp else 0 end as gp, 
            case when t2.g notnull then t2.g else 0 end as g, 
            case when t2.p notnull then t2.p else 0 end as p
            from forward_clusters t1
            left join (select * from skater_stats_career where league_name='NHL') t2 
            on t1.player_id=t2.player_id) 
            group by clusters''')


def cluster_details(k, player_id=0):
    """
    Look up details of cluster k for which the player passed belongs
    :param player_id: player id
    :return: pandas df with all players from given cluster, sorted by career nhl points
    """
    conn = sqlite3.connect('nhl_draft.db')

    cluster = pd.read_sql_query('''
                    select t1.player_id, t1.first_name, t1.last_name, t1.birth_year, t1.g_a17, t1.g_a18, t1.a_a17, t1.a_a18, 
                    case when t2.gp notnull then t2.gp else 0 end as gp, 
                    case when t2.g notnull then t2.g else 0 end as g, 
                    case when t2.p notnull then t2.p else 0 end as p
                    from forward_clusters t1
                    left join (select * from skater_stats_career where league_name='NHL') t2 
                    on t1.player_id=t2.player_id
                    where clusters=:id and t1.player_id <>:pid
                    order by p desc
                    ''', conn, params={'id': k, 'pid': player_id})
    try:
        gp = len(cluster[(cluster['gp'] > 164) & (cluster['birth_year'].astype(int) <= 1994)]) / len(cluster[(cluster['birth_year'].astype(int) <= 1994)])
        p = (cluster[(cluster['gp'] > 164)]['p'].sum() / cluster[(cluster['gp'] > 164)]['gp'].sum()) * 82
        '''
        print(cluster)
        print("Pr(> 164 GP): {0:.2f}".format(gp))
        print("Expected P/82: {0:.2f}".format(p))
        print("Expected Value: {0:.2f}".format(p * gp))
        '''
        return cluster, p * gp, gp, p
    except:
        print("error with cluster {}".format(k))

    return None, None, None, None


def cluster_id(player_id):
    """
    Look up details of cluster k for which the player passed belongs
    :param player_id: player id
    :return: pandas df with all players from given cluster, sorted by career nhl points
    """
    conn = sqlite3.connect('nhl_draft.db')
    cluster = pd.read_sql_query('''
                        select clusters from forward_clusters 
                        where player_id=:id
                        ''', conn, params={'id': player_id})

    try:
        k = int(cluster.iloc[0]['clusters'])
        return k

    except:
        return None


def test(draft_year):
    conn = sqlite3.connect('nhl_draft.db')

    players = pd.read_sql_query('''
                            select t1.player_id 
                            from draft t1
                            inner join bios t2 
                            on t1.player_id = t2.player_id
                            where year=:y and pos2 = "F"
                            ''', conn, params={'y': draft_year})

    player_ids = players['Player_Id'].tolist()
    expected_values = list()
    pr = list()
    p_82 = list()

    for i in player_ids:
        cluster = cluster_details(cluster_id(i), i)
        expected_values.append(cluster[1])
        pr.append(cluster[2])
        p_82.append(cluster[3])

    players['xV'] = expected_values
    players['Pr'] = pr
    players['xP'] = p_82
    players = players.sort_values(by=['xV'], ascending=False)
    players.to_sql('test', conn, if_exists='replace', index=False)

    return players


def logistic_regression(p="F", birth_year=1994):
    conn = sqlite3.connect('nhl_draft.db')

    sample = pd.read_sql_query('''
                    select t1.player_id as player_id, height, weight, age, birth_year,
                    g_a17, g_a18, a_a17, a_a18, case when t2.gp isnull or t2.gp < 164 then 0 else 1 end as gp, 
                    case when t2.gp isnull then 0 else t2.gp end as gp2,
                    case when t2.g_gp isnull then 0 else g_gp end as g_gp,
                    case when t2.a_gp isnull then 0 else a_gp end as a_gp
                    from
                    (select player_id, age, height, weight, birth_year,
                    max(Case when age2 = 17 then normalized_g_gp end) as g_a17,
                    max(Case when age2 = 18 then normalized_g_gp end) as g_a18,
                    max(Case when age2 = 17 then normalized_a_gp end) as a_a17,
                    max(Case when age2 = 18 then normalized_a_gp end) as a_a18
                    from 
                        (select t1.player_id as player_id, age2, age, height, weight, cast(substr(dob,0, 5) as int) as birth_year,
                        sum(gp) as gp, sum(adj_g_gp * le * gp) / sum(gp) as normalized_g_gp, 
                        sum(adj_a_gp * le * gp) / sum(gp) as normalized_a_gp
                        from
                        (select *, t1.g_gp * era_adj_p_gp as adj_g_gp, t1.a_gp * era_adj_p_gp as adj_a_gp, t2.le 
                        from skater_stats_season t1
                        inner join league_equivalencies t2
                        on t1.league_name = t2.league_name
                        inner join era_adjustments t3
                        on t1.league_name = t3.league_name and t1.season = t3.season) t1
                        inner join bios t2 
                        on t1.player_id = t2.player_id
                        where pos2=:p and cast(substr(dob,0, 5) as int)<=:y
                        group by t1.player_id, age2) t1
                    group by player_id) t1
                    left join 
                    (select player_id, gp, g_gp, a_gp
                    from skater_stats_career
                    where league_name = "NHL") t2
                    on t1.player_id = t2.player_id
                    where g_a17 notnull and g_a18 and a_a17 notnull and a_a18 notnull and height notnull and age notnull and weight notnull
                        ''', conn, params={'p': p, 'y': birth_year})

    X = sample[['g_a17', 'g_a18', 'a_a17', 'a_a18']]
    y = sample['gp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

    logistic_regression = LogisticRegression(class_weight='balanced').fit(X_train, y_train)

    sample = sample[sample['gp2'] > 164]

    X = sample[['g_a17', 'g_a18', 'a_a17', 'a_a18', 'age', 'height', 'weight']]
    y = sample[['g_gp', 'a_gp']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

    g_regr = LinearRegression().fit(X_train[['g_a17', 'g_a18']], y_train['g_gp'])
    a_regr = LinearRegression().fit(X_train[['a_a17', 'a_a18']], y_train['a_gp'])
    gpred = g_regr.predict(X_test[['g_a17', 'g_a18']])
    apred = a_regr.predict(X_test[['a_a17', 'a_a18']])
    gtest_set_rmse = (np.sqrt(mean_squared_error(y_test['g_gp'], gpred)))
    gtest_set_r2 = r2_score(y_test['g_gp'], gpred)
    atest_set_rmse = (np.sqrt(mean_squared_error(y_test['a_gp'], apred)))
    atest_set_r2 = r2_score(y_test['a_gp'], apred)
    print(gtest_set_rmse)
    print(gtest_set_r2)
    print(atest_set_rmse)
    print(atest_set_r2)

    test = pd.read_sql_query('''
                        select t1.player_id as player_id, height, weight, age, birth_year,
                        g_a17, g_a18, a_a17, a_a18, case when t2.gp isnull or t2.gp < 164 then 0 else 1 end as gp,
                        case when t2.g_gp isnull then 0 else g_gp end as g_gp,
                        case when t2.a_gp isnull then 0 else a_gp end as a_gp
                        from
                        (select player_id, age, height, weight, birth_year,
                        max(Case when age2 = 17 then normalized_g_gp end) as g_a17,
                        max(Case when age2 = 18 then normalized_g_gp end) as g_a18,
                        max(Case when age2 = 17 then normalized_a_gp end) as a_a17,
                        max(Case when age2 = 18 then normalized_a_gp end) as a_a18
                        from 
                            (select t1.player_id as player_id, age2, age, height, weight, cast(substr(dob,0, 5) as int) as birth_year,
                            sum(gp) as gp, sum(adj_g_gp * le * gp) / sum(gp) as normalized_g_gp, 
                            sum(adj_a_gp * le * gp) / sum(gp) as normalized_a_gp
                            from
                            (select *, t1.g_gp * era_adj_p_gp as adj_g_gp, t1.a_gp * era_adj_p_gp as adj_a_gp, t2.le 
                            from skater_stats_season t1
                            inner join league_equivalencies t2
                            on t1.league_name = t2.league_name
                            inner join era_adjustments t3
                            on t1.league_name = t3.league_name and t1.season = t3.season) t1
                            inner join bios t2 
                            on t1.player_id = t2.player_id
                            where pos2=:p and cast(substr(dob,0, 5) as int)>:y
                            group by t1.player_id, age2) t1
                        group by player_id) t1
                        left join 
                        (select player_id, gp, g_gp, a_gp
                        from skater_stats_career
                        where league_name = "NHL") t2
                        on t1.player_id = t2.player_id
                        where g_a17 notnull and g_a18 and a_a17 notnull and a_a18 notnull and height notnull and age notnull and weight notnull
                            ''', conn, params={'p': p, 'y': birth_year})

    test['pr'] = [i[1] for i in logistic_regression.predict_proba(test[['g_a17', 'g_a18', 'a_a17', 'a_a18']])]
    test['g_gp'] = g_regr.predict(test[['g_a17', 'g_a18']]).tolist()
    test['a_gp'] = a_regr.predict(test[['a_a17', 'a_a18']]).tolist()
    test.to_sql('gp_pr', conn, if_exists='replace', index=False)

    return


if __name__ == '__main__':
    main()
