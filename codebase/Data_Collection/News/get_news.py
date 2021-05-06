from newsapi import NewsApiClient


from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

import pandas as pd


#Creating news api client
newsapi = NewsApiClient(api_key='fcb346126fc44b749798a7e4ae34e54b')

# Get past articles - By default 30 days
def get_past_articles(past=30):
    past_articles = dict()
    for past_days in range(1, past):
        from_day = str((datetime.now() - timedelta(days=past_days)).strftime('%Y-%m-%d'))
        to_day = str((datetime.now() - timedelta(days=past_days -1)).strftime('%Y-%m-%d'))
        past_articles.update({from_day:to_day})
    return past_articles

def get_articles(query, past=30):
    past_articles = get_past_articles(past)
    all_articles = []
    print(past_articles.items())
    for i,j in tqdm(past_articles.items()):
        for pag in tqdm(range(1, 6)):
            pag_articles = newsapi.get_everything(q=query,
                                      from_param=i,
                                      to=j,
                                      language='en',
                                      sort_by='relevancy',
                                      page=pag)['articles']
            if len(pag_articles) == 0: break
            all_articles.extend(pag_articles)
    # print(all_articles)
    return all_articles

# print((get_articles("oneplus", 2)))

#Declare the topic names to fetch the news for

topics = ["oneplus"]

def cleanNewsData(topics):

    df_events = pd.DataFrame(columns=['topic', 'title', 'date', 'desc', 'url', 'content', 'text', 'sources', 'publishedAt'])
    
    for topic in topics:
        
        articles = get_articles(topic, 2)
        titles = [article['title'] for article in articles]
        dates = [article['publishedAt'] for article in articles]
        descriptions = [article['description'] for article in articles]
        urls = [article['url'] for article in articles]
        contents = [article['content'] for article in articles]
        sources = [article['source']['name'] for article in articles]
        publishedAt = [article['publishedAt'] for article in articles]

        df = pd.DataFrame({'title': titles, 'date': dates, 'desc': descriptions, 'url': urls, 'content': contents, 'sources' :sources, 'publishedAt': publishedAt})
        df = df.drop_duplicates(subset='title').reset_index(drop=True)
        df['title'] = df['title'].astype(str)
        df['content'] = df['content'].astype(str)
        df['text'] = df['content'].str[:500]
        df['sources'] =  df['sources'].astype(str)

        df['text'] = df['title'] + '. ' + df['text']
        df['text'] = df['text'].astype(str)
        # df['text'] = df['text'].apply(clean_text)

        df['topic'] = topic
        df = df.dropna()
        #df = df[(df['title'].str.contains(article_filter)) | (df['content'].str.contains(article_filter))]
        # df = df[df['title'].str.contains(article_filter)]

        #print(len(df))
        df_events = df_events.append(df, ignore_index=True)
    
    return df_events

# Use the below line to test the code
# print(cleanNewsData(topics=topics))