import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def main():
    # parse_clusters("D")
    # test(2017)
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
                    select player_id, first_name, last_name, g_a17, g_a18, a_a17, a_a18 from
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
                            where pos2 = :pos and cast(substr(dob,0, 5) as int)<=1994
                            group by t1.player_id, age2) t1
                        group by player_id) 
                    where g_a17 notnull and g_a18 and a_a17 notnull and a_a18 notnull and gp>=20 and dob notnull
                    ''', conn, params={'pos': p})

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

    km = KMeans(n_clusters=50)
    km = km.fit(pre_draft[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']])
    pre_draft.loc[:, 'clusters'] = km.labels_

    pre_draft.to_sql('{}_clusters50'.format(p), conn, if_exists='replace', index=False)

    km = KMeans(n_clusters=100)
    km = km.fit(pre_draft[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']])
    pre_draft.loc[:, 'clusters'] = km.labels_

    pre_draft.to_sql('{}_clusters100'.format(p), conn, if_exists='replace', index=False)

    km = KMeans(n_clusters=200)
    km = km.fit(pre_draft[['g_a17', 'g_a18', 'g_a18', 'a_a17', 'a_a18', 'a_a18']])
    pre_draft.loc[:, 'clusters'] = km.labels_

    pre_draft.to_sql('{}_clusters200'.format(p), conn, if_exists='replace', index=False)

    cluster_details = cur.executescript('''create table {}_clusters as
            select t1.player_id as player_id, t1.clusters as clusters50, t2.clusters as clusters100, 
            t3.clusters as clusters200
            from {}_clusters50 t1
            left join {}_clusters100 t2
            on t1.player_id=t2.player_id
            left join {}_clusters200 t3
            on t1.player_id=t3.player_id'''.format(p, p, p, p))

    drop = cur.executescript('''
                        DROP TABLE IF EXISTS {}_clusters50;'''.format(p))
    drop = cur.executescript('''
                            DROP TABLE IF EXISTS {}_clusters100;'''.format(p))
    drop = cur.executescript('''
                            DROP TABLE IF EXISTS {}_clusters200;'''.format(p))

    # drop = cur.executescript('''
    #                     DROP TABLE IF EXISTS f_clusters2;''')
    #
    # cluster_details = cur.executescript('''create table f_clusters2 as
    #     select clusters, count(player_id) as n, sum(gp)/count(player_id) as gp,
    #     sum(g)/count(player_id) as g, sum(p)/count(player_id) as p/*,
    #     case when sum(g)/sum(gp) isnull then 0 else round(sum(g),2)/round(sum(gp),2) end as g_gp,
    #     case when sum(p)/sum(gp) isnull then 0 else round(sum(p),2)/round(sum(gp),2) end as p_gp*/
    #     from (select t1.*,
    #     case when t2.gp notnull then t2.gp else 0 end as gp,
    #     case when t2.g notnull then t2.g else 0 end as g,
    #     case when t2.p notnull then t2.p else 0 end as p
    #     from f_clusters t1
    #     left join (select * from skater_stats_career where league_name='NHL') t2
    #     on t1.player_id=t2.player_id)
    #     group by clusters''')


# def cluster_details(k, player_id=0):
#     """
#     Look up details of cluster k for which the player passed belongs
#     :param player_id: player id
#     :return: pandas df with all players from given cluster, sorted by career nhl points
#     """
#     conn = sqlite3.connect('nhl_draft.db')
#
#     cluster = pd.read_sql_query('''
#                     select t1.player_id, t1.first_name, t1.last_name, t1.g_a17, t1.g_a18, t1.a_a17, t1.a_a18,
#                     case when t2.gp notnull then t2.gp else 0 end as gp,
#                     case when t2.g notnull then t2.g else 0 end as g,
#                     case when t2.p notnull then t2.p else 0 end as p
#                     from f_clusters t1
#                     left join (select * from skater_stats_career where league_name='NHL') t2
#                     on t1.player_id=t2.player_id
#                     where clusters=:id and t1.player_id <>:pid
#                     order by p desc
#                     ''', conn, params={'id': k, 'pid': player_id})
#     try:
#         gp = len(cluster[(cluster['gp'] > 164)]) / len(cluster[(cluster['gp'])])
#         p = (cluster[(cluster['gp'] > 164)]['p'].sum() / cluster[(cluster['gp'] > 164)]['gp'].sum()) * 82
#         '''
#         print(cluster)
#         print("Pr(> 164 GP): {0:.2f}".format(gp))
#         print("Expected P/82: {0:.2f}".format(p))
#         print("Expected Value: {0:.2f}".format(p * gp))
#         '''
#         return cluster, p * gp, gp, p
#     except:
#         print("error with cluster {}".format(k))
#
#     return None, None, None, None
#
#
# def cluster_id(player_id):
#     """
#     Look up details of cluster k for which the player passed belongs
#     :param player_id: player id
#     :return: pandas df with all players from given cluster, sorted by career nhl points
#     """
#     conn = sqlite3.connect('nhl_draft.db')
#     cluster = pd.read_sql_query('''
#                         select clusters from f_clusters
#                         where player_id=:id
#                         ''', conn, params={'id': player_id})
#
#     try:
#         k = int(cluster.iloc[0]['clusters'])
#         return k
#
#     except:
#         return None
#
#
# def test(draft_year):
#     conn = sqlite3.connect('nhl_draft.db')
#
#     players = pd.read_sql_query('''
#                             select t1.player_id
#                             from draft t1
#                             inner join bios t2
#                             on t1.player_id = t2.player_id
#                             where year=:y and pos2 = "F"
#                             ''', conn, params={'y': draft_year})
#
#     player_ids = players['Player_Id'].tolist()
#     expected_values = list()
#     pr = list()
#     p_82 = list()
#
#     for i in player_ids:
#         cluster = cluster_details(cluster_id(i), i)
#         expected_values.append(cluster[1])
#         pr.append(cluster[2])
#         p_82.append(cluster[3])
#
#     players['xV'] = expected_values
#     players['Pr'] = pr
#     players['xP'] = p_82
#     players = players.sort_values(by=['xV'], ascending=False)
#     players.to_sql('test', conn, if_exists='replace', index=False)
#
#     return players


if __name__ == '__main__':
    main()
