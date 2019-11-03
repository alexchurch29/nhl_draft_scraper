import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def main():

    return


def parse_data():
    """
    Generates a table where each row depicts the delta p/gp between age n, n+1 of a player who played in the same league
    in subsequent seasons. This table will be used to map the aging curves for any given league.
    :return:
    """
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

    return


def parse_aging_curves():
    """
    Fits polynomial aging curves for all leagues that exist in db.
    :return:
    """
    conn = sqlite3.connect('nhl_draft.db')

    leagues = pd.read_sql('SELECT league_name FROM aging_curves', conn)
    leagues = leagues['league_name'].unique().tolist()
    # leagues = ['NHL', 'AHL', 'KHL', 'SHL', 'Liiga', 'NLA', 'DEL', 'Allsvenskan', 'SuperElit', 'Jr. A SM-liiga', 'MHL', 'NCAA', 'USHL', 'OHL', 'WHL', 'QMJHL']
    pos = ['F', 'D']

    aging_curves2 = []

    for league in leagues:
        for p in pos:
            aging_curve = pd.read_sql_query(
                "select * from aging_curves where league_name = :name and pos = :pos order by age0 asc", conn,
                params={'name': league, 'pos': p})
            if len(aging_curve.index) > 1:
                aging_curve.age1 = pd.to_numeric(aging_curve.age1, errors='coerce')
                aging_curve.Delta_P_GP = pd.to_numeric(aging_curve.Delta_P_GP, errors='coerce')
                x = aging_curve.loc[:, 'age1']
                y = aging_curve.loc[:, 'Delta_P_GP']
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        z = np.polyfit(x, y, 2)
                    except np.RankWarning:
                        z = np.polyfit(x, y, 1)
                pp = np.poly1d(z)
                row = list()
                row.append(league)
                row.append(p)
                if len(pp.coeffs) == 3:
                    row.append(pp.coeffs[0])
                    row.append(pp.coeffs[1])
                    row.append(pp.coeffs[2])
                else:
                    row.append(0)
                    row.append(pp.coeffs[0])
                    row.append(pp.coeffs[1])
                aging_curves2.append(row)

    aging_curves2 = pd.DataFrame(aging_curves2, columns=['league_name', 'pos', 'x2', 'x1', 'x0'])
    aging_curves2.to_sql('aging_curves2', conn, if_exists='replace', index=False)

    return


def plot_aging_curves():
    """
    Plot and save figures for each aging curve, distinguished by league/pos.
    :param: aging_curve: pd.Dataframe with league, pos, and coefficients for each aging curve
    :return:
    """

    conn = sqlite3.connect('nhl_draft.db')

    # leagues = pd.read_sql('SELECT league_name FROM aging_curves', conn)
    # leagues = leagues['league_name'].unique().tolist()
    leagues = ['NHL', 'AHL', 'KHL', 'SHL', 'Liiga', 'NLA', 'DEL', 'Allsvenskan', 'SuperElit', 'Jr. A SM-liiga', 'MHL',
               'NCAA', 'USHL', 'OHL', 'WHL', 'QMJHL']
    pos = ['F', 'D']

    for league in leagues:
        for p in pos:
            aging_curve = pd.read_sql_query(
                "select * from aging_curves where league_name = :name and pos = :pos order by age0 asc", conn,
                params={'name': league, 'pos': p})
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
                ax = plt.gca()
                ax.set_xlabel('Age')
                ax.set_ylabel('Delta P/GP')
                ax.tick_params(axis='both', which='both', length=0)
                plt.xticks(np.arange(min(x), max(x) + 1, 1))

                for label, xx, yy in zip(labels, x, aging_curve['Delta_PGP']):
                    plt.annotate(
                        label,
                        xy=(xx, yy),
                        fontsize=5.75)

                plt.savefig('plots/aging_curves/' + league + p + '.jpeg')

    return


if __name__ == '__main__':
    main()
