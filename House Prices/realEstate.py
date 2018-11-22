# webscraper for realestate.com.au
# Isaac Tesla 1 / 06 / 2018

from bs4 import BeautifulSoup # library for sifting through html tags
import requests # to get http for beautiful soup
import csv # to open/close/append CSV
from datetime import datetime # to accuractely stamp the CSV file
import os # to check if file exists
import time # for sleep in loop
import pandas as pd # to use for opening CSV into an array

# 20 results per page
#https://www.realestate.com.au/buy/list-1
#https://www.realestate.com.au/buy/list-201

# specify starting url ====change this to a loop for 200 pages once crawler built
allhouses = 1
while allhouses <= 200:
    requestURL = 'https://www.realestate.com.au/buy/list-' + str(allhouses)
    source = requests.get(requestURL).text

    # query the website and parse html
    soup = BeautifulSoup(source, 'lxml')

    # If file exists append, if not, create
    file_exists = os.path.isfile('housePrices.csv')

    # loop to check if file exists, if not, create it and add a header
    if file_exists == 0:
        csv_file = open('housePrices.csv', 'w', newline='') # Create CSV file to store information - note: newline='' removes extra blank line added by python 3
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Agent','Address','Location','Rooms','Bathrooms','CarSpaces','DwellingType','LandSize','Description','HomeLoanCalculatorMonth'])     # add header information to CSV
        print('File Created: housePrices.csv')
    elif file_exists == 1:
        csv_file = open('housePrices.csv', 'a+', newline='') # Create CSV file to store information - note: newline='' removes extra blank line added by python 3
        csv_writer = csv.writer(csv_file)

    counter = 0
    agent = "a"
    for a in soup.find_all('a', href=True):
        #ADD FILTER for only /job/ at start of a['href']
        if a['href'].startswith("/property"):
            while agent != "None":
                full_URL = 'https://www.realestate.com.au/' + str(a['href'])

                source = requests.get(full_URL).text
                # query the website and parse html
                soup_inner = BeautifulSoup(source, 'lxml')

                # grab outer property information
                try:
                    agent = soup.find_all('article', class_='resultBody')[allhouses].text
                except Exception as e:
                    agent = "None"


                # grab inner advertisement information
                try:
                    address = soup_inner.find_all('span', class_='street-address')[counter].text
                except Exception as e:
                    address = None
                print(address,"\n")

                print(agent)
            counter = counter + 1
    allhouses = allhouses + 1
    counter = 0
csv_file.close()
