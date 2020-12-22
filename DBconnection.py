import cx_Oracle as cx
import pandas as pd


"""
Disconnect from VPN
"""
dsnStr = cx.makedsn("tirpitz.ms.mff.cuni.cz", 1511, "jedenact")
print(dsnStr)
db = cx.connect(user="ahmadli", password="aydin206", dsn=dsnStr)
cur = db.cursor()

cur.execute("select * from IMDB")
db.commit()
df = pd.DataFrame(cur.fetchall())
print(df)


for row in cur.execute("select ACTORS from IMDB"):
    print(row)
