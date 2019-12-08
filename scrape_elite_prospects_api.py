"""
Used to query the eliteprospects API in order to build a database of player bios, historical draft info,
and career stats that will all be used as inputs for our ML prediction model
"""

import time
import requests
import json
import sqlite3
import sys
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# import eliteprospects api key
f = open('api_key.txt', 'r')
api_key = f.read()


def main():
    """
    :param: argv[1] is first draft year in range, argv[2] is final draft year in range
    """
    #call_api()

    return


def get_url(url):
    """
    Get response for a given request
    :param url: given url
    :return: response from server
    """

    response = requests.Session()
    retries = Retry(total=30, backoff_factor=.1)
    response.mount('http://', HTTPAdapter(max_retries=retries))

    try:
        response = response.get(url, timeout=60)
        response.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        return None

    return response


def get_draft_json(draft_year, offset):
    """
    Given a year it returns the eliteprospects json data for that draft
    Ex: http://api.eliteprospects.com/beta/drafts?limit=1000&offset=0&filter=draftType.id=1%26year=2016&sort=overall:asc
    &apikey=abcdefghijk123456
    :param draft_year: year of the draft
    :param offset: offset (limit for each request is 1000 records)
    :return: eliteprospects json data for players drafted in that year
    """

    draft_year = str(draft_year)
    offset = str(offset)
    url = 'http://api.eliteprospects.com/beta/drafts?limit=1000&offset={}&filter=draftType.id=1%26year={}&sort=' \
          'overall:asc&apikey={}'.format(offset, draft_year, api_key)

    try:
        response = get_url(url)
        time.sleep(1)
        draft_json = json.loads(response.text)
    except:
        print('Json for draft {} not returned.'.format(draft_year))
        return None

    return draft_json


def get_player_stats(player_id):
    """
    Given a player id it returns the eliteprospects json data for that players career stats by season
    Ex: http://api.eliteprospects.com/beta/players/199898/stats?limit=1000&sort=season.name:asc&apikey=abcdefghijk123456
    :param player_id: player id
    :return: eliteprospects json data for that player's career stats
    """

    player_id = str(player_id)
    url = 'http://api.eliteprospects.com/beta/players/{}/stats?limit=1000&sort=season.name:asc&apikey={}'.format(
        player_id, api_key)

    try:
        response = get_url(url)
        time.sleep(1)
        player_json = json.loads(response.text)
    except:
        print('Json for player {} not returned.'.format(player_id))
        return None

    return player_json


def get_league_players(league_id, season, offset):
    """
    Given a league id and season it returns the eliteprospects json data containing all player ids for that season
    Ex: http://api.eliteprospects.com/beta/leagues/91/playerstats?season=2017-2018&limit=1000&offset=0&fields=player.id&apikey=abcdefghijk123456
    :param league_id: league id
    :param season: season
    :param offset: offset
    :return: eliteprospects json data containing player ids for the given season
    """

    league_id = str(league_id)
    season = str(season)
    offset = str(offset)
    url = 'http://api.eliteprospects.com/beta/leagues/{}/playerstats?season={}&limit=1000&offset={}&fields=player.id&apikey={}'.format(
        league_id, season, offset, api_key)

    try:
        response = get_url(url)
        time.sleep(1)
        season_json = json.loads(response.text)
    except:
        print('Json for league {} and season {} not returned.'.format(league_id, season))
        return None

    return season_json


def parse_draft_json(start_year, end_year):
    """
    Parses the eliteprospects json for a given range of draft years
    :param start_year: first year to include in timeframe
    :param end_year: final year to include in timeframe
    :return: pandas df with player ids, draft year, round, overall draft selection, and team
    """

    players = list()

    while start_year <= end_year:

        offset = 0
        draft_json = get_draft_json(start_year, offset)
        total_count = draft_json['metadata']['totalCount']

        while True:

            for i in range(0, len(draft_json['data'])):
                player = list()
                try:
                    player.append(draft_json['data'][i]['player']['id'])
                    player.append(draft_json['data'][i]['year'][:4])
                    player.append(draft_json['data'][i]['round'])
                    player.append(draft_json['data'][i]['overall'])
                    player.append(draft_json['data'][i]['team']['name'])
                    players.append(player)
                except:
                    player.append('-1')
                    player.append(draft_json['data'][i]['year'][:4])
                    player.append(draft_json['data'][i]['round'])
                    player.append(draft_json['data'][i]['overall'])
                    player.append(draft_json['data'][i]['team']['name'])
                    players.append(player)

            offset += 1000
            if offset > total_count:
                break
            draft_json = get_draft_json(start_year, offset)

        start_year += 1

    draft_data = pd.DataFrame(players, columns=['Player_Id', 'Year', 'Round', 'OV', 'Team'])

    return draft_data


def parse_player_json(player_id):
    """
    Parses the eliteprospects json for a given player's stats
    :param player_id: id of player
    :return: pandas df for player bios and career stats
    """

    bios = list()
    stats = list()
    player_json = get_player_stats(player_id)

    player = list()
    player.append(player_json['data'][0]['player']['id'])
    try:
        player.append(player_json['data'][0]['player']['firstName'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['lastName'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['dateOfBirth'])
    except:
        try:
            player.append(str(player_json['data'][0]['player']['yearOfBirth']))
        except:
            player.append(None)
    try:
        player.append(player_json['data'][0]['player']['country']['name'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['playerPositionDetailed'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['shoots'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['height'])
    except:
        player.append(None)
    try:
        player.append(player_json['data'][0]['player']['weight'])
    except:
        player.append(None)
    try:
        if player_json['data'][0]['player']['playerPositionDetailed'].find('G') != -1:
            player.append('G')
        elif player_json['data'][0]['player']['playerPositionDetailed'].find('D') != -1:
            player.append('D')
        else:
            player.append('F')
    except:
        player.append(None)
    bios.append(player)

    for i in range(0, len(player_json['data'])):
        try:
            if player_json['data'][i]["gameType"] == "REGULAR_SEASON":
                season = list()
                if player_json['data'][0]['player']['playerPositionDetailed'] != 'G':
                    try:
                        season.append(player_json['data'][i]['player']['id'])
                        season.append(player_json['data'][i]['season']['name'])
                        season.append(player_json['data'][i]['team']['name'])
                        season.append(player_json['data'][i]['league']['name'])
                        season.append(player_json['data'][i]['league']['id'])
                        season.append(player_json['data'][i]['GP'])
                        season.append(player_json['data'][i]['G'])
                        season.append(player_json['data'][i]['A'])
                        season.append(player_json['data'][i]['TP'])
                        season.append(player_json['data'][i]['PIM'])
                        try:
                            season.append(player_json['data'][i]['PM'])
                        except:
                            season.append(None)
                        stats.append(season)
                    except:
                        pass
                else:
                    try:
                        season.append(player_json['data'][i]['player']['id'])
                        season.append(player_json['data'][i]['season']['name'])
                        season.append(player_json['data'][i]['team']['name'])
                        season.append(player_json['data'][i]['league']['name'])
                        season.append(player_json['data'][i]['league']['id'])
                        season.append(player_json['data'][i]['GP'])
                        season.append(player_json['data'][i]['GAA'])
                        season.append(player_json['data'][i]['SVP'])
                        stats.append(season)
                    except:
                        pass
        except:
            pass

    return bios, stats


def parse_league_players(league_id, season):
    """
    Parses the eliteprospects json for a given league and season
    :param league_id: league id
    :param season: season
    :return: list of player ids who played during the given season in the given league
    """

    players = list()

    offset = 0
    season_json = get_league_players(league_id, season, offset)
    total_count = season_json['metadata']['totalCount']

    while True:
        for i in range(0, len(season_json['data'])):
            players.append(season_json['data'][i]['player']['id'])
        offset += 1000
        if offset > total_count:
            break
        season_json = get_league_players(league_id, season, offset)

    return players

def call_api():
    """
    Calls the eliteprospects api and parses all data in order to compile database
    :return:
    """
    conn = sqlite3.connect('nhl_draft.db')
    cur = conn.cursor()

    draft = parse_draft_json(sys.argv[1], sys.argv[2])
    draft.to_sql('draft', conn, if_exists='replace', index=False)

    player_queue = draft['Player_Id'].unique().tolist()  # some players get drafted more than once so we need unique
    player_set = set()
    player_league_queue = list()
    player_league_set = set()

    player_bios = list()
    player_stats = list()
    goalie_stats = list()

    while len(player_queue) > 0:
        player_id = player_queue.pop()
        if player_id not in player_set:
            try:
                player = parse_player_json(player_id)
                if int(player[0][0][3][:4]) > 1969:  # set lower limit on birth year
                    player_bios.append(player[0][0])
                    if player[0][0][5] != 'G':
                        for i in range(0, len(player[1])):
                            player_stats.append(player[1][i])
                            if (player[1][i][4], player[1][i][1]) not in player_league_set and \
                                            (player[1][i][4], player[1][i][1]) not in player_league_queue:
                                player_league_queue.append((player[1][i][4], player[1][i][1]))
                                while len(player_league_queue) > 0:
                                    league_season = player_league_queue.pop()
                                    new_players = parse_league_players(league_season[0], league_season[1])
                                    for j in range(0, len(new_players)):
                                        if new_players[j] not in player_queue and new_players[j] not in player_set:
                                            player_queue.append(new_players[j])
                                    player_league_set.add(league_season)
                    else:
                        for i in range(0, len(player[1])):
                            goalie_stats.append(player[1][i])
                player_set.add(player_id)
            except:
                print("Error for player {}".format(player_id))
                player_set.add(player_id)

    player_bios = pd.DataFrame(player_bios,
                               columns=['Player_Id', 'First_Name', 'Last_Name', 'DOB', 'Country', 'Pos', 'Shoots',
                                        'Height', 'Weight', 'Pos2'])
    player_stats = pd.DataFrame(player_stats,
                                columns=['Player_Id', 'Season', 'Team', 'League_Name', 'League_Id', 'GP', 'G', 'A', 'P',
                                         'PIM', '+/-'])
    goalie_stats = pd.DataFrame(goalie_stats,
                                columns=['Player_Id', 'Season', 'Team', 'League_Name', 'League_Id', 'GP', 'GAA', 'SVP'])

    player_bios.to_sql('bios', conn, if_exists='replace', index=False)
    player_stats.to_sql('skater_stats', conn, if_exists='replace', index=False)
    goalie_stats.to_sql('goalie_stats', conn, if_exists='replace', index=False)

    drop_player_season = cur.executescript('''
                    DROP TABLE IF EXISTS skater_stats_season;''')
    drop_player_career = cur.executescript('''
                        DROP TABLE IF EXISTS skater_stats_career;''')
    drop_goalie_season = cur.executescript('''
                            DROP TABLE IF EXISTS goalie_stats_season;''')

    skater_season_stats = cur.executescript('''
                create table skater_stats_season as
                select t1.*, case when length(t2.dob)>4 then round((julianday((substr(t1.season,-4) || "-09-15")) - 
                julianday(dob))/365.25,2) else null end as age, substr(t1.season,-4) - substr(dob,0, 5) as age2, 
                round(round(g,2)/gp,2) as G_GP, round(round(a,2)/gp,2) as A_GP, round(round(P,2)/gp,2) as P_GP
                from skater_stats t1
                inner join bios t2 
                on t1.Player_Id = t2.Player_Id''')

    skater_career_stats = cur.executescript('''
                    create table skater_stats_career as
                    select t1.*, round(round(t1.g,2)/t1.gp,2) as G_GP, round(round(t1.a,2)/t1.gp,2) as A_GP, 
                    round(round(t1.P,2)/t1.gp,2) as P_GP
                    from (select bios.Player_Id, league_name, league_id, sum(gp) as GP, sum(g) as G, sum(a) as A, sum(p) as P
                    from skater_stats
                    inner join bios 
                    on player_stats.Player_Id = bios.Player_Id
                    group by player_stats.Player_Id, league_id) t1''')

    goalie_season_stats = cur.executescript('''
                    create table goalie_stats_season as
                    select t1.*, round((julianday((substr(t1.season,-4) || "-09-15")) - julianday(dob))/365.25,2) as age,
                    substr(t1.season,-4) - substr(dob,0, 5) as age2
                    from goalie_stats t1
                    inner join bios t2 
                    on t1.Player_Id = t2.Player_Id
                    where length(t2.dob)>4''')

    drop_player_temp = cur.executescript('''
                DROP TABLE IF EXISTS skater_stats;''')
    drop_goalie_temp = cur.executescript('''
                DROP TABLE IF EXISTS goalie_stats;''')


def parse_player_accolades():
    """
    Parse player accolades from eliteprospects player page in order to build table containing awards and other accolades
    http://api.eliteprospects.com/beta/players/6146?apikey=abcdefghijk123456
    :return:
    """
    conn = sqlite3.connect('nhl_draft.db')

    nhl = pd.read_sql_query('''select player_id from skater_stats_career where league_name = "NHL"
                                union all 
                                select distinct(t1.player_id) from goalie_stats_season t1 
                                inner join bios t2 
                                on t1.player_id=t2.player_id
                                where league_name = "NHL" and pos2 = "G"''', conn)
    player_ids = nhl['Player_Id'].tolist()
    df = list()

    for i in range(0, len(player_ids)):
        player_id = player_ids[i]
        url = 'http://api.eliteprospects.com/beta/players/{}?apikey={}'.format(player_id, api_key)
        try:
            response = get_url(url)
            time.sleep(1)
            player_json = json.loads(response.text)
        except:
            print('Json for player {} not returned.'.format(player_id))
            df.append([player_id, 0, 0, 0, 0, 0, 0])
            continue
        # id, hart(7464), norris(7467), vezina(7471), first all-star(230), second all-star(149), all-star(223)
        awards = [player_id, 0, 0, 0, 0, 0, 0]
        for j in player_json['data']['awards']:
            try:
                if j['awardType']['id'] == 7464:
                    awards[1] += 1
                if j['awardType']['id'] == 7467:
                    awards[2] += 1
                if j['awardType']['id'] == 7471:
                    awards[3] += 1
                if j['awardType']['id'] == 230:
                    awards[4] += 1
                if j['awardType']['id'] == 149:
                    awards[5] += 1
                if j['awardType']['id'] == 223:
                    awards[6] += 1
            except:
                pass
        df.append(awards)

    player_awards = pd.DataFrame(df,
                                columns=['Player_Id', 'Hart', 'Norris', 'Vezina', 'First_All_Star', 'Second_All_Star', 'All_Star'])

    player_awards.to_sql('awards', conn, if_exists='replace', index=False)

    add_wjcs = pd.read_sql_query('''select t1.player_id as player_id, WJC_18, WJC_20, 
                                Hart, Norris, Vezina, First_All_Star, Second_All_Star, All_Star
                                from (select distinct(player_id) as player_id
                                from skater_stats_season
                                union all 
                                select distinct(player_id) as player_id
                                from goalie_stats_season) t1 
                                left join (select player_id, count(player_id) as WJC_18
                                from skater_stats_season
                                where league_name = "WJC-18"
                                group by player_id
                                union all 
                                select player_id, count(player_id) as WJC_18
                                from goalie_stats_season
                                where league_name = "WJC-18"
                                group by player_id) t2
                                on t1.player_id=t2.player_id
                                left join (select player_id, count(player_id) as WJC_20
                                from skater_stats_season
                                where league_name = "WJC-20" and age2 <= 18
                                group by player_id
                                union all
                                select player_id, count(player_id) as WJC_20
                                from goalie_stats_season
                                where league_name = "WJC-20" and age2 <= 18
                                group by player_id) t3
                                on t1.player_id=t3.player_id
                                left join awards t4
                                on t1.player_id = t4.player_id''', conn)

    add_wjcs.to_sql('awards', conn, if_exists='replace', index=False)

    return


if __name__ == '__main__':
    main()

f.close()
