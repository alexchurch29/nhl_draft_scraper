import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def main():
    conn = sqlite3.connect('nhl_draft.db')
    cur = conn.cursor()

    aging_curves = cur.executescript('''
                    create table aging_curves as
                    select t1.league_name0 as league_name, t1.pos2 as pos, t1.n as n, t1.age0 as age0, t1.age1 as age1, sum(t2.delta_p_gp) as Delta_P_GP
                    from 
                    (select league_name0, pos2, age0, age1, count(player_id) as n, 
                    ifnull( round (((round(SUM(P1),4)/round(sUM(GP1),4)) - (round(SUM(P0),4)/round(sUM(GP0),4))) / (round(SUM(P0),4)/round(sUM(GP0),4)), 2), 0) as Delta_P_GP

                    from (
                        Select t1.player_id, t3.pos2, t1.season as season0, t2.season as season1, t1.league_name as league_name0, t2.league_name as league_name1,
                                    t1.age2 as Age0, t2.age2 as Age1, t1.P as P0, t2.P as P1, t1.GP as GP0, t2.GP as GP1
                                    FROM skater_stats_season t1
                                    cross join skater_stats_season t2
                                    on t1.player_id = t2.player_id
                                    inner join bios t3 
                                    on t1.player_id = t3.player_id
                                    where t1.age2 = (t2.age2 -1)
                        ) t1
                    where league_name0=league_name1
                    group by league_name0, age0, age1, pos2
                    )t1
                    inner join 
                    (select league_name0, pos2, age0, age1, count(player_id) as n, 
                    ifnull( round (((round(SUM(P1),4)/round(sUM(GP1),4)) - (round(SUM(P0),4)/round(sUM(GP0),4))) / (round(SUM(P0),4)/round(sUM(GP0),4)), 2), 0) as Delta_P_GP

                    from (
                        Select t1.player_id, t3.pos2, t1.season as season0, t2.season as season1, t1.league_name as league_name0, t2.league_name as league_name1,
                                    t1.age2 as Age0, t2.age2 as Age1, t1.P as P0, t2.P as P1, t1.GP as GP0, t2.GP as GP1
                                    FROM skater_stats_season t1
                                    cross join skater_stats_season t2
                                    on t1.player_id = t2.player_id
                                    inner join bios t3 
                                    on t1.player_id = t3.player_id
                                    where t1.age2 = (t2.age2 -1)
                        ) t1
                    where league_name0=league_name1
                    group by league_name0, age0, age1, pos2
                    )t2
                    on t1.league_name0=t2.league_name0 and t1.pos2=t2.pos2 and t1.age0 >= t2.age0
                    where t1.n >= 10
                    group by t1.league_name0, t1.pos2, t1.age0, t1.age1
                    ''')

    leagues = pd.read_sql('SELECT league_name FROM aging_curves', conn)
    leagues = leagues['league_name'].unique().tolist()
    # leagues = ['NHL', 'AHL', 'KHL', 'SHL', 'Liiga', 'NLA', 'DEL', 'Allsvenskan', 'SuperElit', 'Jr. A SM-liiga', 'MHL', 'NCAA', 'USHL', 'OHL', 'WHL', 'QMJHL']
    pos = ['F', 'D']

    for league in leagues:
        for p in pos:
            aging_curve = pd.read_sql_query("select * from aging_curves where league_name = :name and pos = :pos order by age0 asc", conn, params={'name': league, 'pos': p})
            if len(aging_curve.index) > 1:
                aging_curve.age1 = pd.to_numeric(aging_curve.age1, errors='coerce')
                aging_curve.Delta_P_GP = pd.to_numeric(aging_curve.Delta_P_GP, errors='coerce')
                x = aging_curve.loc[:, 'age1']
                y = aging_curve.loc[:, 'Delta_P_GP']
                labels = aging_curve.loc[:, 'n']
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        z = np.polyfit(x, y, 2)
                    except np.RankWarning:
                        z = np.polyfit(x, y, 1)
                pp = np.poly1d(z)
                aging_curve['Delta_PGP'] = pp(x)
                aging_curve.plot(x='age1', y='Delta_PGP', title=league + ' ' + p, legend=False, label='n')
                for label, xx, yy in zip(labels, x, aging_curve['Delta_PGP']):
                    plt.annotate(
                        label,
                        xy=(xx, yy),
                        fontsize=5.75)
                plt.show()


if __name__ == '__main__':
    main()
