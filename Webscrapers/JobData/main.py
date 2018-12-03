#!/usr/bin/env python3
#Isaac Tesla SIT742 Assessment 2 Main file

#import libraries
import platform
import webScrape.py
import aggregateData.py

#declare main loop
def main():
    #run the webscraper - note this can take around 2 hours
    #webScrape.call(["python", "webScrape.py"])
    aggregateData.call(["python", "aggregateData.py"])


#run main program
if __name__ == '__main__': main()
