SELECT * FROM IMDB;
SELECT * FROM IMDB;
SELECT * FROM USER_SYNONYMS;
SELECT * FROM ALL_SYNONYMS;
--DROP synonym syn_name;

-- !! CREATING synonym --
-- My synonyms -> A_MMOVIES, A_MUSERS, A_MRATINGS
--create synonym A_MRATINGS for ruleml.MRATINGS;


-------------------------------------------------  QUERY ---------------------------------------------------------------
-- !! 1st Query --
select ACTORS from IMDB;
select ACTORS from IMDB WHERE ROWNUM <= 50;
--select ACTORS from IMDB FETCH NEXT 1 ROWS ONLY;

-- !! 2nd Query --
select count(*) from A_MRATINGS;
select USERID, MOVIEID, RATING from A_MRATINGS where rownum <=10000;

-- !! 3rd Query --
select * from IMDB;
select XML from IMDB where TT = 'tt3591516'
select XML from IMDB
select A_MMOVIES.TT, MOVIEID, XML from A_MMOVIES inner join IMDB on A_MMOVIES.TT = IMDB.TT


--------------------------------------------  XML Handling -------------------------------------------------------------

-- !! Not Runnable in Python (because of SET long 5000 - no worries, we already got CLOB from below query)
SET long 5000 SELECT e.XML.getClobval() AS coXML FROM IMDB e where rownum <=6

-- !! Runnable in Python (got XMLs as CLOB objects)
SELECT e.TT, e.XML.getClobval() AS coXML, A_MMOVIES.MOVIEID  FROM IMDB e inner join A_MMOVIES on e.TT = A_MMOVIES.TT where rownum <= 100

-- !! Runnable: short version of above query
SELECT e.XML.getClobval() AS coXML FROM IMDB e where TT = 'tt3591516'
