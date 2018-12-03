# Webscraper to collect all SEEK.com.au data
# typically 1000 pages of job advertisements, with 22 ads per page
# Start URL = https://www.seek.com.au/jobs?page=1
# last URL = https://www.seek.com.au/jobs?page=1000
# Isaac Tesla - SIT742 Assessment 2

# import libraries
from bs4 import BeautifulSoup # library for sifting through html tags
import requests # to get http for beautiful soup
import csv # to open/close/append CSV
from datetime import datetime # to accuractley stamp the CSV file
import os # to check if file exists


# specify starting url ====change this to a loop for 1000 pages once crawler built
all_seek = 1
while all_seek <= 1000:
    requestURL = 'https://www.seek.com.au/jobs?page=' + str(all_seek)
    source = requests.get(requestURL).text

    # query the website and parse html
    soup = BeautifulSoup(source, 'lxml')

    # If file exists append, if not, create
    file_exists = os.path.isfile('seek_data.csv')

    # loop to check if file exists, if not, create it and add a header
    if file_exists == 0:
        csv_file = open('seek_data.csv', 'w', newline='') # Create CSV file to store information - note: newline='' removes extra blank line added by python 3
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Main_Job_Title','Job_Title', 'Date_Advertised','Location', 'Work_Type', 'Classification', 'Job_Description', 'Page URL'])     # add header information to CSV
        print('File Created: seek_data.csv')
    elif file_exists == 1:
        csv_file = open('seek_data.csv', 'a+', newline='') # Create CSV file to store information - note: newline='' removes extra blank line added by python 3
        csv_writer = csv.writer(csv_file)

    counter = 0
    for a in soup.find_all('a', href=True):
        #ADD FILTER for only /job/ at start of a['href']
        if a['href'].startswith("/job/"):
            full_URL = 'http://www.seek.com.au' + str(a['href'])
            #print(full_URL)

            source = requests.get(full_URL).text
            # query the website and parse html
            soup_inner = BeautifulSoup(source, 'lxml')

            # grab outer Job_Title
            Main_Job_Title = soup.find_all('a', class_='_1EkZJQ7')[counter].text
            counter = counter + 1

            # get job information; if no job title...keep going
            try:
                Job_Title = soup_inner.find('h1', class_='jobtitle').text.strip()
            except Exception as e:
                Job_Title = None #since we want it to keep going but not output the error message

            try:
                Date_Advertised = soup_inner.findAll('span', class_='lwHBT6d')[1].text
            except Exception as e:
                Date_Advertised = None

            try:
                Location = soup_inner.findAll('strong', class_='lwHBT6d')[3].text
                if Location == "right to live and work" or Location == " right to live and work" or Location == " right to live and work ":
                    Location = soup_inner.findAll('strong', class_='lwHBT6d')[4].text
            except Exception as e:
                Location = None

            try:
                Work_Type = soup_inner.findAll('span', class_='lwHBT6d')[2].text
                if Work_Type == "Part Time" or Work_Type == "Casual/Vacation" or Work_Type == "Contract/Temp" or Work_Type == "Full Time":
                    Work_Type = soup_inner.findAll('span', class_='lwHBT6d')[2].text
                else:
                    Work_Type = soup_inner.findAll('span', class_='lwHBT6d')[3].text
            except Exception as e:
                Work_Type = None

            try:
                Classification = soup_inner.findAll('strong', class_='lwHBT6d')[4].text
                if Classification == Location:
                    # change to the actual classification
                    Classification = soup_inner.findAll('strong', class_='lwHBT6d')[5].text
            except Exception as e:
                Classification = None

            try:
                Job_Description = soup_inner.find('div', class_='templatetext').text.strip()
            except Exception as e:
                Job_Description = None

            # write all scraped information to CSV
            csv_writer.writerow([Main_Job_Title, Job_Title, Date_Advertised, Location, Work_Type, Classification, Job_Description, full_URL])

            print('Main page:', all_seek, '; item', counter)
    all_seek = all_seek + 1
    counter = 0
csv_file.close()

print('File Updated')
