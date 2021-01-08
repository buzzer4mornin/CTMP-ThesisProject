import cx_Oracle as cx
import pandas as pd
import os

os.chdir("..")


def get_credentials() -> list:
    c = []
    with open('credentials.txt') as f:
        for line in f.readlines():
            try:
                # fetching username and password
                _, value = line.split(": ")
            except:
                # raises error
                print('Add your username and password in credentials file')
                exit(0)
            c.append(value.rstrip(" \n"))
    return c


"-*- Disconnect from VPN -*-"
dsnStr = cx.makedsn("tirpitz.ms.mff.cuni.cz", 1511, "jedenact")
print(dsnStr)
db_credentials = get_credentials()

try:
    # Connect to DB
    db = cx.connect(*db_credentials, dsn=dsnStr)
    cur = db.cursor()

    # -- 1st Query --  [Get Actors table]
    '''# cur.execute("select ACTORS from IMDB WHERE ROWNUM <= 50")
    # db.commit()
    # df = pd.DataFrame(cur.fetchall())
    # df.columns = ["Actors"]'''

    # -- 2nd Query --  [Get Ratings table]
    '''cur.execute("select USERID, MOVIEID, RATING from A_MRATINGS")
    db.commit()
    df = pd.DataFrame(cur.fetchall())
    df.columns = ["USERID", "MOVIEID", "RATING"]
    df["RATING"] = df['RATING'].apply(lambda x: 1 if x >= 4 else 0)
    df.to_pickle(str(os.path.dirname(os.path.abspath(__file__))) + '/rating_df')'''

    # -- 3rd Query --  [Get XML table from IMDB]
    '''cur.execute(
        "SELECT e.TT, e.XML.getClobval() AS coXML, A_MMOVIES.MOVIEID  FROM IMDB e inner join A_MMOVIES on e.TT = A_MMOVIES.TT")
    db.commit()
    df = cur.fetchall()
    # columns: TT(str) -- CLOB(obj) -- MOVIEID(int)
    print(df)
    plot_extractor = lambda xml: (xml.split('plot="'))[1].split('"')[0]
    for i in range(len(df)):
        df[i] = list(df[i])
        df[i][1] = plot_extractor(df[i][1].read())
        print(i)
    df = pd.DataFrame(df)
    df.columns = ["TT", "MOVIEPLOT", "MOVIEID"]
    df.to_pickle(str(os.path.dirname(os.path.abspath(__file__))) + '/movie_df')'''

    # Another way to Query
    # for row in cur.execute("select ACTORS from IMDB"):
    #    print(row)

    print("Table Created successful")

except cx.DatabaseError as e:
    print("ERROR", e, "| or check VPN connection!")

else:
    # Close all when done
    if cur: cur.close()
    if db: db.close()


