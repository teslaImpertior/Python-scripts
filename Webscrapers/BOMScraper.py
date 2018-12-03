#!/usr/bin/env python

#
#Get the most recent station list from: ftp://ftp.bom.gov.au/anon2/home/ncc/metadata/sitelists/stations.zip
#This script only registers the stations that are currently active.
#For best results, use this script on the last day of the month.
#

import urllib.request, urllib.error
import sqlite3
import os
import datetime

# This scraper grabs all BOM weather data available in the last 14 months. Enjoy.

#I love this name.
class StationToDatabase():

    '''
        Stations file layout:
            Site [0] - Name [1] - Lat [2] - Lon [3] - Start [4,5] - End [6,7] - Years [8] - % [9] - Obs [10] - AWS [11]
    '''

    def __init__(self, stations_file='stations.txt', db_file='data.db'):
        self.station_file = stations_file
        self.db_conn = self.__initDB(db_file) #Initialize connection.
        self.month, self.year = self.getDate()

    def __initDB(self, db_file):
        #Check for existing DB file.
        file_exists = False
        if os.path.isfile(db_file):
            file_exists = True
        conn = sqlite3.connect(db_file)
        if not file_exists:
            try:
                #Initialize the database structure.
                c = conn.cursor()
                sql = 'CREATE TABLE stations(\
                stationID CHAR(6) NOT NULL,\
                stationName CHAR(40) NOT NULL,\
                stationOperating INT NOT NULL,\
                stationScanned INT NOT NULL);'
                c.execute(sql)
                conn.commit()
                sql = 'CREATE TABLE data(\
                stationID CHAR(6) NOT NULL,\
                dataDate DATE NOT NULL,\
                minTemp REAL NOT NULL,\
                maxTemp REAL NOT NULL,\
                rainFall REAL NOT NULL);'
                c.execute(sql)
                conn.commit()
            except:
                conn.rollback()
                print("Something went wrong while initializing the DB.")
        return conn

    def close(self):
        self.db_conn.close()
        print("DB connection closed.")

    def getDate(self):
        cur_date = str(datetime.datetime.now()).split()[0]
        cur_date = cur_date.split('-')
        year = str(cur_date[0])
        month = str.lower(str(datetime.datetime.now().strftime("%B")))[:3]
        return year, month

    #Load data from file and send to DB.
    def load(self):
        with open(self.station_file, 'rU') as f:
            print("Loading content.")
            lst_content = f.readlines()
            f.close()
            print("Got content, transferring to database...")
            for line in lst_content:
                self.addData(line, 0)

    #Clean the data before inserting into DB
    def clean_data(self, data, dType):
        clean_data = []
        #station data
        if dType == 0:
            data = data.split()
            #print(data)
            i = 0
            #Station number
            stat_num = data[i]
            while len(stat_num) < 6:
                stat_num = '0' + stat_num
            clean_data.append(stat_num)
            i += 1
            #Station name
            stat_name = data[i]
            #print("stat_name: {0} at i = {1}".format(stat_name, i))
            i += 1
            #Basically incrementing until it finds a float value. Seems the easiest way to get the name of each station.
            while True:
                try:
                    float(data[i])
                    i += 1
                    break
                except ValueError:
                    stat_name += ' ' + data[i]
                    #print("stat_name: {0} at i = {1}".format(stat_name, i))
                    i += 1
            stat_name = stat_name.replace("'", " ")
            clean_data.append(stat_name)
            i += 3
            #Currently operating?
            cur_operating = '0'
            month = data[i].lower()
            if month == self.month:
                year = data[i+1]
                if year == self.year:
                    cur_operating = '1'
            clean_data.append(cur_operating)
        #climate data
        elif dType == 1:
            #',2016-03-1,15.6,23.4,0,,,SSW,41,14:43,18.5,73,,SSE,15,1020.9,23.0,46,,SSW,20,1018.2'
            data = data.split(',')
            #Date
            date = data[1]
            #Temps
            minTemp = data[2]
            if minTemp == '':
                minTemp = '-1'
            maxTemp = data[3]
            if maxTemp == '':
                maxTemp = '-1'
            #Rain
            rainFall = data[4]
            if rainFall == '':
                rainFall = '-1'
            #Append to result list.
            clean_data.append(date)
            clean_data.append(minTemp)
            clean_data.append(maxTemp)
            clean_data.append(rainFall)
        return clean_data

    #Add a line of data to the DB.
    def addData(self, data, dType, station=None):
        #station data
        if dType == 0:
            try:
                lst_data = self.clean_data(data, dType)
                c = self.db_conn.cursor()
                sql = "INSERT INTO stations (stationID,stationName,stationOperating,stationScanned) VALUES ('{0}','{1}','{2}',0)".format(lst_data[0], lst_data[1], lst_data[2])
                c.execute(sql)
                self.db_conn.commit()
            except sqlite3.Error as er:
                self.db_conn.rollback()
                print("Error inserting station data: ", er)
        #climate data
        # ',2016-03-1,15.6,23.4,0,,,SSW,41,14:43,18.5,73,,SSE,15,1020.9,23.0,46,,SSW,20,1018.2'
        elif dType == 1:
            try:
                if station == None:
                    raise Exception('Station not made available.')
                #print(data)
                lst_data = self.clean_data(data, dType)
                #print(lst_data)
                c = self.db_conn.cursor()
                sql = "INSERT INTO data (stationID,dataDate,minTemp,maxTemp,rainFall) VALUES ('{0}','{1}','{2}','{3}','{4}')".format(station, lst_data[0], lst_data[1], lst_data[2], lst_data[3])
                c.execute(sql)
                #print("SQL ran fine.")
                self.db_conn.commit()
                #print("Inserted data for {0}".format(station))
            except sqlite3.Error as er:
                self.db_conn.rollback()
                print("Error: ", er)
                #print("Error inserting climate data: {0}".format(station))
        return True

    def displayCounts(self):
        c = self.db_conn.cursor()
        sql = "SELECT count(*) FROM stations"
        c.execute(sql)
        num = c.fetchone()
        print("stations table has {0} entries.".format(num[0]))
        c = self.db_conn.cursor()
        sql = "SELECT count(*) FROM stations WHERE stationOperating = 1"
        c.execute(sql)
        num = c.fetchone()
        print("There are currently {0} active stations.".format(num[0]))
        sql = "SELECT count(*) FROM data"
        c.execute(sql)
        num = c.fetchone()
        print("data table has {0} entries.".format(num[0]))

    def readAllData(self):
        with open('station_output.txt', 'w') as f:
            c = self.db_conn.cursor()
            sql = "SELECT * FROM stations"
            c.execute(sql)
            print("Printing stations.")
            for d in c.fetchall():
                myStr = ""
                for dd in d:
                    myStr += str(dd) + " "
                print(myStr)
                f.write(myStr + "\n")
        with open('data_output.txt', 'w') as f:
            print("Printing climate data.")
            c = self.db_conn.cursor()
            sql = "SELECT * FROM data"
            c.execute(sql)
            for d in c.fetchall():
                myStr = ""
                for dd in d:
                    myStr += str(dd) + " "
                print(myStr)
                f.write(myStr + "\n")
            f.close()

    def updateStationScanned(self, station):
        try:
            c = self.db_conn.cursor()
            sql = "UPDATE stations SET stationScanned = '1' WHERE stationID = '{0}'".format(station)
            c.execute(sql)
            self.db_conn.commit()
        except sqlite3.Error as er:
            self.db_conn.rollback()
            print("SQLite3 Error: ", er)

    def addClimateData(self, debug=None):

        c = self.db_conn.cursor()
        sql = "SELECT stationID FROM stations WHERE stationOperating = 1 AND stationScanned = 0"
        c.execute(sql)
        dr = DataRetriever()
        if debug == None:
            for s in c.fetchall():
                lst_data = dr.retrieve(s[0])
                if not lst_data == None:
                    for d in lst_data:
                        d = d.split('\\r\\n')
                        for dd in d:
                            self.addData(dd, 1, station=s[0])
                    self.updateStationScanned(s[0])
                    print("Successfully processed station {0}".format(s[0]))
        else:
            for s in debug:
                lst_data = dr.retrieve(s)
                if not lst_data == None:
                    for d in lst_data:
                        d_s = d.split('\\r\\n')
                        for dd in d_s:
                            #print(d_s)
                            self.addData(dd, 1, station=s)
                    print("Successfully processed station {0}".format(s))
        print("Finished.")
        #for s in c.fetchall():
        #    dr.retrieve(int(s[0]))

class DataRetriever():

    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.19 (KHTML, like Gecko) Ubuntu/12.04 Chromium/18.0.1025.168 Chrome/18.0.1025.168 Safari/535.19'
        self.urlregex = "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        self.month, self.year = self.getDate()

    def retrieve(self, station):
        #URL using station number: http://www.bom.gov.au/jsp/ncc/cdio/weatherData/av?p_nccObsCode=201&p_display_type=dwo&p_startYear=&p_c=&p_stn_num=000000
        #First task is to get the first/latest CSV file. --> /climate/dwo/201603/text/IDCJDW6002.201603.csv
        #                                                --> /climate/dwo/201603/html/IDCJDW6002.201603.shtml
        lst_data = []
        bom_domain = "http://www.bom.gov.au"
        url = self.resolve_redirects("{0}/jsp/ncc/cdio/weatherData/av?p_nccObsCode=201&p_display_type=dwo&p_startYear=&p_c=&p_stn_num={1}".format(bom_domain, station))
        #print(url)
        if url == None:
            return None
        else:
            sid, start_date = self.getParams(url)
            lst_dates = self.getDates(start_date)
            #print(lst_dates)
            for date in lst_dates:
                cur_url = '{0}/climate/dwo/{1}/text/{2}.{3}.csv'.format(bom_domain, date, sid, date)
                #print("Retrieving {0}".format(cur_url))
                try:
                    with urllib.request.urlopen(urllib.request.Request(cur_url, headers={'User-Agent': self.user_agent})) as page:
                        raw_content = str(page.read())
                        page.close()
                        d = raw_content.split('"')[-1].replace("'"," ")[4:]
                        lst_data.append(d)
                        print("Retrieved {0}".format(cur_url))
                except urllib.error.HTTPError:
                    print("HTTPError")
                    pass
                except urllib.error.URLError:
                    print("Encountered a urllib error.")
                    pass
            return lst_data

        #http://www.bom.gov.au/climate/dwo/201603/html/IDCJDW6002.201603.shtml
        #http://www.bom.gov.au/climate/dwo/201603/text/IDCJDW6002.201603.csv
        #sid = IDCJDW6002

    def getDates(self, start_date):
        lst_dates = []
        year = int(start_date[:4])
        month = int(start_date[4:])
        #Only 14 months of data publicly available.
        i = 0
        while i < 14:
            if month == 0:
                year -= 1
                month += 12
            if month < 10:
                m = '0{0}'.format(str(month))
                lst_dates.append('{0}{1}'.format(year, m))
            else:
                lst_dates.append('{0}{1}'.format(year, month))
            month -= 1
            i += 1
        return lst_dates

    #Get the url-modifies station id.
    def getParams(self, url):
        sid = ""
        d = url.split('/')[5]
        sid = d.split('.')[0]
        date = self.year + self.month
        return sid, date

    def getDate(self):
        cur_date = str(datetime.datetime.now()).split()[0]
        cur_date = cur_date.split('-')
        year = cur_date[0]
        month = cur_date[1]
        return year, month

    #urllib does not redirect on its own, this is a quick fix.
    def resolve_redirects(self, url):
        target_url = "http://www.bom.gov.au"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': self.user_agent})) as page:
                raw_content = str(page.read())
                page.close()
            #Because the content of this page is always a handful of bytes, we can be as lazy as we want.
            path = raw_content.split(' URL=')[1]
            path = path.split('"')[0]
            target_url += path
        except IndexError:
            print("Could not resolve a new URL for {0}.".format(url))
            return None
        except: #urllib.error:
            print("There was an error with urllib in resolve_redirects()")
            return None
        return target_url

#Simply uncomment the functionality that you want to run, command line will come later.
STD = StationToDatabase(stations_file="stations.txt", db_file="data.db")
#STD.load()             #Gets and loads station data.
#STD.addClimateData()   #Iterates through each of the available stations and gets the climate data.
#STD.readAllData()      #Read out all data to the command line/IDLE.
#STD.displayCounts()     #Display a count of entires in each table.
STD.close()
