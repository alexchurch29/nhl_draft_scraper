import sqlite3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def main():

    conn = sqlite3.connect('nhl_draft.db')

    leagues = pd.read_sql('''
                    select league_name0, league_name1, n, LE from (
                    select league_name0, league_name1, age0, age1, count(player_id) as n, 
                    ifnull( round ((round(SUM(P1),4)/round(sUM(GP1),4)) / (round(SUM(P0),4)/round(sUM(GP0),4)), 2), 0) as LE
                    from(
                    Select t1.player_id, t3.pos2, t1.season as season0, t2.season as season1, t1.league_name as league_name0, t2.league_name as league_name1,
                                t1.GP as GP0, t2.GP as GP1, t1.age2 as Age0, t2.age2 as Age1, t1.G as G0, t2.G as G1, 
                                t1.A as A0, t2.A as A1, t1.P as P0, t2.P as P1 
                                FROM skater_stats_season t1
                                cross join skater_stats_season t2
                                on t1.player_id = t2.player_id
                                inner join bios t3 
                                on t1.player_id = t3.player_id
                                where t1.age2 = (t2.age2 - 1) /*and cast(substr(t1.season,5,-4) as integer) >= 2008*/ and t1.gp >= 20 and t2.gp >= 20
                    )
                    group by league_name0, league_name1)
                    where n >= 20''', conn)

    league0 = leagues['league_name0'].tolist()
    league1 = leagues['league_name1'].tolist()
    obs = leagues['n'].tolist()
    equivalencies = leagues['LE'].tolist()
    league_list = list(set(league0 + league1))

    G = nx.DiGraph()
    G.add_nodes_from(league_list)
    for i in range(0, len(league1)):
        G.add_edge(league0[i], league1[i], n=obs[i], LE=equivalencies[i])

    league_equivalencies = list()
    for i in range(0, len(league_list)):
        wLE = list()
        for path in nx.all_simple_paths(G, source=league_list[i], target='NHL', cutoff=3):
            prod = 1
            for j in range(0, len(path)-1):
                prod *= G.edge[path[j]][path[j+1]]['LE']
            wLE.append((prod, len(path)))

        LE = 0
        for k in range(0, len(wLE)):
            LE += wLE[k][0]/wLE[k][1]**2
        if len(wLE) > 0:
            LE = LE / len(wLE)
        league_equivalencies.append([league_list[i], LE])

    league_equivalencies = pd.DataFrame(league_equivalencies, columns=['league_name', 'LE'])
    NHL = league_equivalencies.loc[league_equivalencies["league_name"] == "NHL"]["LE"].values[0]
    league_equivalencies["LE"] = league_equivalencies["LE"].apply(lambda x: x / NHL)
    equivalencies = league_equivalencies['LE'].tolist()
    league_equivalencies = league_equivalencies.sort_values(by=["LE"], ascending=False)
    league_equivalencies.to_sql('league_equivalencies', conn, if_exists='replace', index=False)

    league_list = ['NHL', 'AHL', 'KHL', 'SHL', 'Liiga', 'NLA', 'DEL', 'Allsvenskan', 'SuperElit', 'Jr. A SM-liiga', 'MHL', 'NCAA', 'USHL', 'OHL', 'WHL', 'QMJHL']
    labels = dict()
    for i in range(0, len(league_list)):
        labels[league_list[i]] = league_list[i]

    # plt.figure(3, figsize=(90, 90))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    nx.draw_networkx(G=G, pos=nx.spring_layout(G, k=0.5, iterations=20), node_list=G.nodes(), node_color='orange', node_size=[i * 1000 for i in equivalencies], edge_color='blue', alpha=0.2, arrows=False, font_size=7, labels=labels)
    plt.savefig('plots/league_equivalencies.jpeg')


if __name__ == '__main__':
    main()
