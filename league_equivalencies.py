import sqlite3

conn = sqlite3.connect('nhl_draft.db')
cur = conn.cursor()

skater_season_stats = cur.executescript('''
            create table league_equivalencies as

            select league_name0, league_name1, pos2, age1, count(player_id) as n, 
            round ((round(SUM(P1),4)/round(sUM(GP1),4)) / (round(SUM(P0),4)/round(sUM(GP0),4)), 2) as LE
            from(
            Select t1.player_id, t3.pos2, t1.season as season0, t2.season as season1, t1.league_name as league_name0, t2.league_name as league_name1,
                        t1.GP as GP0, t2.GP as GP1, t1.age2 as Age0, t2.age2 as Age1, t1.G as G0, t2.G as G1, 
                        t1.A as A0, t2.A as A1, t1.P as P0, t2.P as P1 
                        FROM skater_stats_season t1
                        cross join skater_stats_season t2
                        on t1.player_id = t2.player_id
                        inner join bios t3 
                        on t1.player_id = t3.player_id
                        where t1.season <> t2.season and (t1.age2 = (t2.age2 - 1))
            )
            group by league_name0, league_name1, pos2, age1
            order by league_name1, pos2, age1 asc''')
