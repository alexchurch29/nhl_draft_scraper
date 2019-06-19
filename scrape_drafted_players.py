import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup


def get_url(url):
    """
    Get the url
    :param url: given url
    :return: raw html
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


def get_urls_single_year(draft_year):
    """
    Given a draft_year it returns the player page urls of all players drafted that year
    :param draft_year: e.g. 2000
    :return: list of player page urls
    """

    draft_year = str(draft_year)
    url = 'https://www.eliteprospects.com/draft/nhl-entry-draft/{}'.format(draft_year)

    try:
        html = get_url(url)
        time.sleep(1)
    except Exception as e:
        print('HTML for draft year {} is not there'.format(draft_year), e)
        raise Exception

    soup = BeautifulSoup(html.content, 'html.parser')
    players = soup.find_all(class_="player")
    players = [str(player.find('a')) for player in players]
    players = [player.split('"')[1] for player in players if player.find('href="') != -1]

    return players


def get_urls_multi_year(start, stop):
    """
    Given a range of draft_years it returns the player page urls of all players drafted between the specified years
    :param start: start year
    :param stop: stop year
    :return: list of player urls
    """

    players = list()

    for i in range(start, stop + 1):
        players += get_urls_single_year(i)

    return players


def player_profile(url):
    """
    Given a player page url scrapes the player profile for bio info and career stats
    :param url: player page url
    :return: list with player info
    """

    try:
        html = get_url(url)
        time.sleep(1)
    except Exception as e:
        print('HTML for player {} is not there'.format(url), e)
        raise Exception

    soup = BeautifulSoup(html.content, 'html.parser')

    player = soup.find_all(class_='table-view')
    player = [x.get_text().split('\n') for x in player][0]
    player = [x.strip() for x in player if x != '']

    try:
        id = url.split('/')[4]
        name = url.split('/')[-1].replace('-', ' ').title()
        dob = player[player.index('Date of Birth')+1]
        country = player[player.index('Nation')+1]
        pos = player[player.index('Position') + 1]
        height = player[player.index('Height') + 1][-6:]
        weight = player[player.index('Weight') + 1][:7]

        draft = player[player.index('Drafted') + 1].split(' ')
        d_year = draft[0]
        d_pos = draft[3][1:]

        team = draft[draft.index('by')+1:]
        d_team = ''
        for i in team:
            d_team += i
            d_team += ' '
        d_team = d_team.strip()

        player_bio = list()
        player_bio.append(id)
        player_bio.append(name)
        player_bio.append(dob)
        player_bio.append(country)
        player_bio.append(pos)
        player_bio.append(height)
        player_bio.append(weight)
        player_bio.append(d_year)
        player_bio.append(d_pos)
        player_bio.append(d_team)
        bio = list()
        bio.append(player_bio)

        player_bio = pd.DataFrame(bio, columns=['id', 'Name', 'DOB', 'Country', 'Pos', 'Height', 'Weight', 'Draft_Year', 'Draft_Pos', 'Draft_Team'])

        player = soup.find("table", "table table-striped table-condensed table-sortable player-stats highlight-stats")

        stats = list()
        for line in player.findAll('tr'):
            for l in line.findAll('td'):
                stats.append(l.getText().replace('\n', '').strip())

        if pos != 'G':
            stats = [stats[i:i + 17][:9] for i in range(0, len(stats), 17) if stats[i:i + 17][5] != '-']

            for i in stats:
                if i[0] == '':
                    i[0] = None
                else:
                    i[0] = i[0].replace('-', '')

            stats = pd.DataFrame(stats, columns=['Season', 'Team', 'League', 'GP', 'G', 'A', 'P', 'PIM', '+/-'])

        else:
            stats = [stats[i:i + 11][:6] for i in range(0, len(stats), 11) if stats[i:i + 11][5] != '-']

            for i in stats:
                if i[0] == '':
                    i[0] = None
                else:
                    i[0] = i[0].replace('-', '')

            stats = pd.DataFrame(stats, columns=['Season', 'Team', 'League', 'GP', 'GAA', 'SV%'])

        stats['Season'].fillna(method='ffill', inplace=True)
        stats.insert(loc=0, column='id', value=id)
    except:
        return

    return player_bio, stats, pos


def parse_multi_year(start, end):

    urls = get_urls_multi_year(start, end)

    bios = []
    player_stats = []
    goalie_stats = []

    for i in urls:
        player = player_profile(i)
        try:
            bios.append(player[0])
            if player[2] != 'G':
                player_stats.append(player[1])
            else:
                goalie_stats.append(player[1])
        except:
            continue

    player_bios = pd.concat(bios, ignore_index=True)
    player_stats = pd.concat(player_stats, ignore_index=True)
    goalie_stats = pd.concat(goalie_stats, ignore_index=True)

    pd.DataFrame.to_csv(player_bios, 'bios.csv', index=False)
    pd.DataFrame.to_csv(player_stats, 'player_stats.csv', index=False)
    pd.DataFrame.to_csv(goalie_stats, 'goalie_stats.csv', index=False)

    return

parse_multi_year(1995, 2018)
