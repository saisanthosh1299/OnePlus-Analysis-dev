# -*- coding: utf-8 -*-

#Need to make changes in the code 
# Need to parametarize

from lxml import html  
import requests
import pandas as pd
import time
import timeit
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s')

## Samsung Galaxy s9

# reviewsDF = pd.DataFrame()
#  https://www.gsmarena.com/samsung_galaxy_s9-reviews-8966p2.php

def gsm_scraper( brand, model, url, pages):

    logging.info('%s AT Model', model)

    reviewsDF = pd.DataFrame()

    for i in range(1,pages):

        gsm_url = url+'p'+str(i)+'.php'
        # print(gsm_url)
        logging.info('%s AT Model', gsm_url)
        logging.info('%s At page and %s Pages', i, pages)
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'

        headers = {'User-Agent': user_agent}
        page = requests.get(gsm_url, headers = headers)
        parser = html.fromstring(page.content)

    #    xpath_reviews = '//div[@class="user-thread"]'
    #    reviews = parser.xpath(xpath_reviews)

        xpath_msg  = './/p[@class="uopin"]//text()'
        msg = parser.xpath(xpath_msg)

        review_dict = {'body': msg}
        reviewsDF['page'] = i
        reviewsDF['url'] = gsm_url
        reviewsDF = reviewsDF.append(review_dict, ignore_index=True)
        print(i)
        time.sleep(10)

    reviewsDF['brand'] = brand
    reviewsDF['model'] = model
    reviewsDF['pages'] = pages

    return reviewsDF

# append to csv 
# reviewsDF.to_csv(r'tweets_s9.csv', header=None, index=None, sep=' ', mode='a')

def get_reviews(file = "gsm_input.csv"):
    
    result_df = pd.DataFrame()

    df_input = pd.read_csv(file)
    # df_input = pd.read_excel(file, engine='openpyxl')
    
    # print(df_input)
    for i in df_input.index:
        # print(i)
        brand = df_input['brand'][i]
        model = df_input['model'][i]
        url = df_input['url'][i]
        pages = df_input['pages'][i]
        
        df = gsm_scraper(brand = brand, model = model, url = url, pages = pages)
        if len(df) == 0:
            logging.info('%s skipped URL', url)
        else:
            result_df = result_df.append(df)
    
    result_df.to_csv("gsm_reviews_data.csv")

#Use for test purposes
# get_reviews(file= "gsm_input_sample.csv")

# %timeit
get_reviews()
