
import pandas as pd

#For fetching youtube video transcript
from youtube_transcript_api import YouTubeTranscriptApi

def getTranscript(videoId):

    #Checking if the text transcript for a video is available or not
    try:
        videoTranscript = YouTubeTranscriptApi.get_transcript(videoId, languages=['en'])
    except:
        return pd.DataFrame()

    if len(videoTranscript) != 0:
        #Construct a DF from the video transcript
        #Converting list to dataframe
        transcriptDF = pd.DataFrame(videoTranscript)

        #Adding a new column for transcript reference
        transcriptDF["VideoId"] = videoId
    else:
        return pd.DataFrame()

    return transcriptDF

df_videos = pd.read_csv("videos.csv")

result_df = pd.DataFrame()

for video_id in df_videos['video']:
    # print(video_id)
    df = getTranscript(video_id)
    if len(df) == 0:
        print(video_id)
    else:
        result_df = result_df.append(df)

result_df.to_csv("results.csv")



# print(getTranscript("VHt0LqGZVSY"))