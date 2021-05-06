import os




# class TwitterConfig:
#     CONSUMER_KEY = os.environ.get('CONSUMER_KEY')
#     CONSUMER_SECRET = os.environ.get('CONSUMER_SECRET')
#     ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
#     ACCESS_TOKEN_SECRET = os.environ.get('ACCESS_TOKEN_SECRET')

class TwitterConfig:
    consumer_key = 'uGTILkgCip1rW0xAZGFI4wrZd'
    consumer_secret = 'DXWuuvGHSpx4nTVbDLQ6eV4MWo2WYGRwuv80HhWSv62uyZYzY1'
    access_token = '719501289420238848-8zrxpZRg6WqAfVAuPUozJpMzm5HVOLT'
    access_token_secret= 'Kg1qo5lwiX9OQDHENf6hnpANUfu4ZjLqRWVwTMrGkjuhY'
    
    CONSUMER_KEY = consumer_key
    CONSUMER_SECRET = consumer_secret
    ACCESS_TOKEN = access_token
    ACCESS_TOKEN_SECRET = access_token_secret


# class DBConfig:
#     USER = os.environ.get('DB_USER')
#     PWORD = os.environ.get('DB_PWORD')
#     HOST = os.environ.get('DB_HOST')

class DBConfig:
    USER = "postgres"
    PWORD = "root"
    HOST = "localhost:5432"

if __name__ == '__main__':
    print("Started")