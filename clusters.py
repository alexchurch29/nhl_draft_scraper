import sqlite3
import pandas as pd
from sklearn.cluster import KMeans


def main():

    return


def parse_time_series_data(p="F"):
    conn = sqlite3.connect('nhl_draft.db')
    cur = conn.cursor()

    pre_draft = pd.read_sql_query('''
            select * from
                (select player_id, first_name, last_name,
                max(Case when age = 17 then normalized_p_gp end) as a17,
                max(Case when age = 18 then normalized_p_gp end) as a18
                from 
                    (select t1.player_id as player_id, age2 as age, first_name, last_name, sum(p_gp * le * gp) / sum(gp) as normalized_p_gp
                    from
                    (select *, t2.le from skater_stats_season t1
                    inner join league_equivalencies t2
                    on t1.league_name = t2.league_name) t1
                    inner join bios t2 
                    on t1.player_id = t2.player_id
                    where pos2 = :pos
                    group by t1.player_id, age2) t1
                group by player_id) 
            where a17 notnull and a18 notnull
            ''', conn, params={'pos': p})

    '''
    Sum_of_squared_distances = []
    K = range(100, 350, 50)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(pre_draft[['a17', 'a18']])
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    '''
    km = KMeans(n_clusters=350)
    km = km.fit(pre_draft[['a17', 'a18']])
    pre_draft.loc[:, 'clusters'] = km.labels_

    if p == "F":
        pre_draft.to_sql('forward_clusters', conn, if_exists='replace', index=False)

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


def cluster_details(k):
    conn = sqlite3.connect('nhl_draft.db')
    cluster = pd.read_sql_query('''
                    select t1.player_id, t1.first_name, t1.last_name, t1.a17, t1.a18, 
                    case when t2.gp notnull then t2.gp else 0 end as gp, 
                    case when t2.g notnull then t2.g else 0 end as g, 
                    case when t2.p notnull then t2.p else 0 end as p
                    from forward_clusters t1
                    left join (select * from skater_stats_career where league_name='NHL') t2 
                    on t1.player_id=t2.player_id
                    where clusters=:id
                    order by p desc
                    ''', conn, params={'id': k})

    return cluster

if __name__ == '__main__':
    main()
