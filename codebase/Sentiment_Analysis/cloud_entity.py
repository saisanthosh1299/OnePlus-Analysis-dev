from google.cloud import language
from cleantext import clean
import pandas as pd

def clean_text(text):

    cleaned_text = clean(text,
                            fix_unicode=True,               # fix various unicode errors
                            to_ascii=True,                  # transliterate to closest ASCII representation
                            lower=True,                     # lowercase text
                            no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,                  # replace all URLs with a special token
                            no_emails=False,                # replace all email addresses with a special token
                            no_phone_numbers=True,         # replace all phone numbers with a special token
                            no_numbers=False,               # replace all numbers with a special token
                            no_digits=False,                # replace all digits with a special token
                            no_currency_symbols=False,      # replace all currency symbols with a special token
                            no_punct=False,                 # remove punctuations
                            replace_with_punct="",          # instead of removing punctuations you may replace them
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number="<NUMBER>",
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"                       # set to 'de' for German special handling
                        )
    print(cleaned_text)
    return cleaned_text

#Replace the key file
def analyze_text_entities(text):
    client = language.LanguageServiceClient.from_service_account_json("C:\\Users\\Srinu\\Documents\\keys\\nlp.json")
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.analyze_entities(document=document,encoding_type = 'UTF32' )
    print(response.entities)
    result = []
    for entity in response.entities:
        
        if entity.type_.name == "DATE":
            # print("=" * 80)
            result.append(entity.name)
            # results = dict(
            #     name=entity.name,
            #     # type=entity.type_.name,
            #     # salience=f"{entity.salience:.1%}",
            #     # wikipedia_url=entity.metadata.get("wikipedia_url", "-"),
            #     # mid=entity.metadata.get("mid", "-"),
            # )
            # for k, v in results.items():
            #     print(f"{k:15}: {v}")
    print(result)
    return result


def get_entities(file):

    result_df = pd.DataFrame()

    df_input = pd.read_csv(file)
    # df_input = pd.read_excel(file, engine='openpyxl')

    cols = ['row_id', 'date_list','cleaned_text','body','brand','model']
    result_df = pd.DataFrame(columns = cols)
    print(result_df)
    for i in df_input.index:
        # print(i)
        
        text = df_input['body'][i]
        
            
        brand = df_input['brand'][i]
        model = df_input['model'][i]

        text_cleaned = clean_text(text)

        entities_list = analyze_text_entities(text_cleaned)
            
        result_df.loc[i] = [i, entities_list, text_cleaned, text, brand, model]
            
            # result_df['row_id'].append(i)
            # result_df['date_list'][i] = entities_list
            # result_df['cleaned_text'][i] = text_cleaned
            # result_df['body'][i] = text
            # result_df['brand'][i] = brand
            # result_df['model'][i] = model
    
    return result_df

get_entities("D:\\ampba\\fp1\\codebase\\clone\\codebase\\Data_Collection\\News\\gsm_reviews_data.csv").to_csv("gsmarena_entities_v1.csv")


# text = "Phonekidd 24 Nov 2020 Bruh I still use a galaxy S5 Still Using Ace 2 S2S4 Note 4  This S9 Old Is Gold Root Ace 2 Lag  More Slower Than The Stock So I Flash Back Stock  It Still Feel Slow But Better Then Root The Rest Still Work Amazing Especially S9 This Blazing Fast  Excellent Camera Great Speed  Gaming Performance Battery Life Also Great Android 10 Supported Custom Support For Android 11 Via Project Treble Or Android 10 Custom Rom Jawir750 19 Nov 2020 November 2020 update has been released Now Wait For December 2020 Patch MS 29 Nov 2020 Put a new battery to S8 you can use it for 2 more years Idk if i can even find one these days  I already upgraded to an S10 a year ago and now im just using the S8 as a backup so its not really a big deal  AnonD919675 25 Nov 2020 Thats impressive rnThe S5 in todays standards isnt even mid range level so you mi  more Put a new battery to S8 you can use it for 2 more years Phonekidd 24 Nov 2020 Bruh I still use a galaxy S5 Thats impressive   rnThe S5 in todays standards isnt even mid range level so you might want to upgrade if you havent but the fact that it still works is nice  rnIve never had my Samsung phones last more than 3 years and my S8 is now pretty close to death with the phone randomly turning off and the battery only lasting for about 40 mins before going to 0 AnonD919675 23 Nov 2020 Im surprised when i see someone say still using the S9 today and it works goodquo  more Bruh I still use a galaxy S5 Im surprised when i see someone say still using the S9 today and it works good   rnLike bruh this phone is only 2 years old If youre still using an S5 then id be surprised but not this  rnWhen you pay 1000 for a phone you better hope it lasts you atleast 34 years otherwise youd just be wasting that money if you had to change your phone every 12 years  Just a regular guy 14 Nov 2020 I bought my unlocked phone from eBay from a highly rated seller 2 years ago samsungs u  more Send it for me then  November 2020 update has been released Got ny s9 two years ago and still its sleek and lovely without any crack or damege though was dropped 2 3 times   rnPerformance is excellent and never lagged and stucked with all lits of apps Super fast and does everything like a charm and in fact basically i can do all online works with this lovely s9 Still its supports all the modern sharing methods and standard technologies Camera is also good Music is excellent  with loud sorround sound sterio speakers Battery one day only but thats enough for me Peronalisations are immense with lots of sertings   No any problems occurred so far Android sent ver 10 for this and now gives only security updates but who cares i love my sleek and high performance  s9  To improve my  64gb storage  i fixed 128gb sd card Love you s9  I bought my unlocked phone from eBay from a highly rated seller 2 years ago samsungs ui 25 update reset the phone to Sprints lease option lock screen cannot use any more I guess samsung doesnt want you to use unlocked phones anymore  rn My samsung s9 got google etc in the background like a watermark Not getting a reason or sollution from samsung Sn 02 Nov 2020 Can we receive it or not I have it too for few days now central EU arksmc05 28 Oct 2020 got one ui 25 in india today  Can we receive it or not Got ONE UI 25 update on October 31  rn Excepting battery lifethis phone is still goodcpu its a beast comparing many other phonesvery good screencamerastereo soundpremium materialsstill looking good for 2020so i keep using itstill no reason to change got one ui 25 in india today  I really love this phone After 2 years its still so fast The camera has been better after updates Now it has portrait and night modes too The screen with no notch or a strange black hole in the middle of it I wish these type of phones would come back Headphone jack via 35 mm and a cleanpowerful output Stereo speakers also The only thing is 3000 mah battery the rest of it for me is just quite perfect The phone is great no doubt about it I specially like the curved screen and it feels good to hold Using this phone since it was launched in 2018 But recently I am having this screen blackout issue when trying to lock and unlock suddenly The simple fix is to unlock the phone 5 seconds after locking it Rest everything is fine with the phone Have googled about this and a lot of users seem to have this issue Little disappointed with this but in no mood to replace until it becomes unusable Same with the update ONE UI 21 the phone is faster and looks much better overall only bad thing battery"

# analyze_text_entities(text)