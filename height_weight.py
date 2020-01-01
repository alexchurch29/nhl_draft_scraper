import time
import sqlite3
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from joblib import dump, load
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scrape_elite_prospects_api import get_url


def main():
    run_regression("height")
    return


def parse_html(draft_year):
    """
    parse data for specified draft year
    :param draft_year: e.g. 2000
    :return: df with all relevant data
    """

    draft_year = str(draft_year)
    url = 'http://www.nhl.com/ice/draftstats.htm?year={}&team=&position=S&round='.format(draft_year)

    try:
        html = get_url(url)
        time.sleep(1)
    except Exception as e:
        print('HTML for draft year {} is not there'.format(draft_year), e)
        raise Exception

    soup = BeautifulSoup(html.content, 'html.parser')

    data = []
    table = soup.find('table', attrs={'class': 'data'})
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [i.text.strip() for i in cols]
        data.append(cols)
    data = [i for i in data if i and i[0][:5] != 'Round']

    for i in data:
        height = i[7]
        if len(height) == 6:
            i[7] = int(((int(height[0]) * 12) + int(height[3:5])) * 2.54)
        if len(height) == 5:
            i[7] = int(((int(height[0]) * 12) + int(height[3])) * 2.54)
        i[8] = int(int(i[8]) * 0.453592)

    data = pd.DataFrame(data, columns=['Round', 'Pick', 'OV', 'Team', 'Name', 'Pos', 'Country', 'Height', 'Weight', 'League_Name', 'Junior_Team'])
    data['Year'] = draft_year

    return data


def parse_draft_range(start_year, end_year):
    """
    concatenate dataframes for specified range and save to db
    :param start_year: start year
    :param end_year: end year
    :return: n/a
    """
    dfs = []
    for i in range(start_year, end_year + 1):
        dfs.append(parse_html(i))
    data = pd.concat(dfs)

    conn = sqlite3.connect('nhl_draft.db')
    data.to_sql('height_weight', conn, if_exists='replace', index=False)
    return


def run_regression(var="height"):
    """
    run regression to extroplate heights and weights for prospects
    :param var: height or weight
    :return: saves a fitted linear regression model to extrapolate a prospect's height/weight at age 25+
    """
    conn = sqlite3.connect('nhl_draft.db')

    df = pd.read_sql_query('''
                    select t1.player_id, case when length(t2.dob)>4 then round((julianday((t1.year || "-09-15")) - julianday(dob))/365.25,2) else null end as age0, 
                    t3.height as height0, t3.weight as weight0, t2.height as height1, t2.weight as weight1, t2.pos2 as pos
                    from draft t1
                    inner join bios t2
                    on t1.player_id = t2.player_id
                    inner join height_weight t3
                    on t1.year = t3.year and t1.OV = t3.OV
                    where t1.year between 2014 and 2018 and t2.height notnull and t2.weight notnull and pos2 notnull
                    ''', conn)

    df['Pos2'] = np.where(df.pos == "D", 1, 0)

    X = df[['age0', 'height0', 'weight0', 'Pos2']]
    y = df[['{}1'.format(var)]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Accuracy:", model.score(X_test, y_test))

    save_model(model, 'models/{}.sav'.format(var))

    return


def save_model(clf, filename):
    '''
    save a trained model to be used later on
    :param clf: classifier to save
    :param filename: filename
    :return: None
    '''
    dump(clf, filename)

    return

if __name__ == '__main__':
    main()
