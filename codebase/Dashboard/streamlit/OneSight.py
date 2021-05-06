import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import warnings

import sys
import os
from  PIL import Image

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
# from config import DBConfig
import datetime
import time

#plotly related
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# st.set_option('wideMode', True)

max_width_str = f"max-width: 2000px;"
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{{max_width_str}}}
    </style>""", unsafe_allow_html=True
)
pd.options.plotting.backend = "plotly"

logo = Image.open("..//assets//logo.png")

#############Read The Data
@st.cache()
def get_website_data():
    df = pd.read_csv("..//data//website_traffic_v1.csv")
    df_new = df.groupby(['month', 'year'])['oneplus', 'samsung'].sum().reset_index()
    df_new['Year_Month'] = df_new['year'].astype(str) + "-" + df_new['month'].astype(str)
    df_new.sort_values(by =['year', 'month'], inplace = True)
    return df_new

@st.cache()
def get_benchmark():
    df = pd.read_csv("..//data//benchmark.csv")
    return df

@st.cache()
def get_youtube():
    df = pd.read_csv("..//data//results_per_video_v2.csv")
    mean = df["sentiment"].mean( skipna = True)
    return mean

# @st.cache()
def get_gsm_entities():
    df = pd.read_csv("..//data//gsmarena_entities_v3.csv")
    return df

####################
def filter_data(df, min, max):
    pass

def groupby_year(df,min,max):
    pass
#title & favicon
# st.set_page_config(page_title='One Sight', page_icon = logo, layout = 'wide', initial_sidebar_state = 'auto')

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.sidebar.image(logo, use_column_width = False, width = 150 )
# st.sidebar.title("<center>Welcome to OneSight")
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Welcome to OneSight 💡</h1>", unsafe_allow_html=True)

###Get Gauge Chart
def get_gauge( title, delta, value = 50):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': delta},
        gauge = {'axis': {'range': [None, 100]},
                    'threshold' : {'line': {'color': "green", 'width': 10}, 'thickness': 0.75, 'value': value}})
        )
    fig.update_layout( paper_bgcolor='rgb(233,233,233)')
    return fig

# @st.cache()
def gsm_entities_viz(brand = "samsung", minYear = 2016, maxYear =2020 ):
    # competitor
    df = get_gsm_entities()
    df['NewSentiment'] = df['sentiment']/10
    df['Newmagnitude'] = df['magnitude']/10
    # opinion part
    df = df[['brand','model','NewSentiment','Newmagnitude','Month','Year','MonthNuMBER']]
    df_op = df.loc[(df['brand'] == brand) ]
    df_op1 = df_op.groupby(['Month', 'Year','MonthNuMBER'])['NewSentiment','Newmagnitude'].mean().reset_index()
    df_op1 = df_op1.loc[(df_op1['Year'] >= minYear) & (df_op1['Year'] <= maxYear)]
    df_op1.sort_values(by =['Year', 'MonthNuMBER'], inplace = True)
    df_op1['Year_Month'] = df_op1['Year'].astype(str) + "-" + df_op1['MonthNuMBER'].astype(str)
    df_op1 = df_op1[['Year_Month','Newmagnitude','NewSentiment']]
    return df_op1
    
def o_gsm_entities_viz(brand = "oneplus", minYear = 2016, maxYear=2020, model="8"):

    df = get_gsm_entities()
    df['NewSentiment'] = df['sentiment']/10
    df['Newmagnitude'] = df['magnitude']/10
    # opinion part
    df = df[['brand','model','NewSentiment','Newmagnitude','Month','Year','MonthNuMBER']]
    df_op = df.loc[(df['brand'] == brand) & (df['model'] == model) ]
    df_op1 = df_op.groupby(['Month', 'Year','MonthNuMBER'])['NewSentiment','Newmagnitude'].mean().reset_index()
    df_op1 = df_op1.loc[(df_op1['Year'] >= minYear) & (df_op1['Year'] <= maxYear)]
    df_op1.sort_values(by =['Year', 'MonthNuMBER'], inplace = True)
    df_op1['Year_Month'] = df_op1['Year'].astype(str) + "-" + df_op1['MonthNuMBER'].astype(str)
    df_op1 = df_op1[['Year_Month','Newmagnitude','NewSentiment']]
    return df_op1

############# Twitter Analyser###############
def twitter_analyser():

    with st.spinner("**Sit Back**, OneSight is reasoning 🧠 "):
        time.sleep(1)
    st.markdown("<h2 style='text-align: center; color: black;'>Twitter Analyser📨</h2>", unsafe_allow_html=True)
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def get_con():
        USER = "postgres"
        PWORD = "root"
        HOST = "localhost:5432"
        return create_engine('postgresql://{}:{}@{}/postgres'.format(USER, PWORD, HOST),
                            convert_unicode=True)


    @st.cache(allow_output_mutation=True, show_spinner=False, ttl=5*60)
    def get_data():
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        df = pd.read_sql_table('tweets', get_con())
        df = df.rename(columns={'body': 'Tweet', 'tweet_date': 'Timestamp',
                                'followers': 'Followers', 'sentiment': 'Sentiment',
                                'keyword': 'Subject'})
        return df, timestamp


    @st.cache(show_spinner=False)
    def filter_by_date(df, start_date, end_date):
        df_filtered = df.loc[(df.Timestamp.dt.date >= start_date) & (df.Timestamp.dt.date <= end_date)]
        return df_filtered


    @st.cache(show_spinner=False)
    def filter_by_subject(df, subjects):
        return df[df.Subject.isin(subjects)]


    @st.cache(show_spinner=False)
    def count_plot_data(df, freq):
        plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).id.count().unstack(level=0, fill_value=0)
        plot_df.index.rename('Date', inplace=True)
        plot_df = plot_df.rename_axis(None, axis='columns')
        return plot_df


    @st.cache(show_spinner=False)
    def sentiment_plot_data(df, freq):
        plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).Sentiment.mean().unstack(level=0, fill_value=0)
        plot_df.index.rename('Date', inplace=True)
        plot_df = plot_df.rename_axis(None, axis='columns')
        return plot_df


    

    data, timestamp = get_data()

    # st.header('Twitter Analyser')
    # st.markdown("<h1 style='text-align: center; color: black;'>Twitter Analyser📨{}</h1>", unsafe_allow_html=True)
    st.write('Total tweet count: **{}**'.format(data.shape[0]))
    st.write('Data last loaded {} (In GMT +0 Timezone)'.format(timestamp))

    # col1, col2 = st.beta_columns(2)

    date_options = data.Timestamp.dt.date.unique()
    start_date_option = st.sidebar.selectbox('Select Start Date', date_options, index=0)
    end_date_option = st.sidebar.selectbox('Select End Date', date_options, index=len(date_options)-1)

    keywords = data.Subject.unique()
    keyword_options = st.sidebar.multiselect(label='Subjects to Include:', options=keywords.tolist(), default=keywords.tolist())

    data_subjects = data[data.Subject.isin(keyword_options)]
    data_daily = filter_by_date(data_subjects, start_date_option, end_date_option)

    top_daily_tweets = data_daily.sort_values(['Followers'], ascending=False).head(10)

    plot_freq_options = {
        'Hourly': 'H',
        'Four Hourly': '4H',
        'Daily': 'D'
    }
    plot_freq_box = st.sidebar.selectbox(label='Plot Frequency:', options=list(plot_freq_options.keys()), index=0)
    plot_freq = plot_freq_options[plot_freq_box]

    st.subheader('Tweet Volumes')
    plotdata = count_plot_data(data_daily, plot_freq)
    st.line_chart(plotdata)

    st.subheader('Sentiment')
    plotdata2 = sentiment_plot_data(data_daily, plot_freq)
    st.line_chart(plotdata2)

    
    st.subheader('Influential Tweets')
    st.dataframe(top_daily_tweets[['Tweet', 'Timestamp', 'Followers', 'Subject']].reset_index(drop=True), 1000, 400)

    st.subheader('Recent Tweets')
    st.table(data_daily[['Tweet', 'Timestamp', 'Followers', 'Subject']].sort_values(['Timestamp'], ascending=False).
                reset_index(drop=True).head(10))


    # locations = pd.DataFrame(pd.eval(data_daily[data_daily['location'].notnull()].location), columns=['lon', 'lat'])
    # st.map(locations)

#############################################

#############Competitive Analysis############

# @st.cache(suppress_st_warning=True)
def competitive_analysis():
    with st.spinner("**Sit Back**, OneSight is reasoning 🧠 "):
        time.sleep(1)
    selected_date_range = st.sidebar.slider('Select the Date Range', 2016, 2020, (2018, 2020), 1)
    min,max = selected_date_range[0],selected_date_range[1]
    # st.header("Competative Analysis")
    st.markdown("<h2 style='text-align: center; color: black;'><b>Competitive Analysis 📉📈<b></h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>Viewing Data from {} to {}  </h4>".format(min, max), unsafe_allow_html=True)
    # st.write("Viewing Data from " + str(min) + " to " + str(max))
    value=80
    # delta=56
    title1="Oneplus"
    title2="Samsung"
    st.markdown("<h4 style='text-align: center; color: black;'>Youtube📽️ Expert Reviews Score</h1>", unsafe_allow_html=True)
    col1, col2 = st.beta_columns(2)
    # gauge_fig = make_subplots(rows = 1, cols = 2)
    
    col1, col2 = st.beta_columns(2)
    
    
    guage_mean = get_youtube()
    gauge_fig_1 = go.Figure()
    gauge_fig_1.add_trace(
        go.Indicator(
        mode = "gauge+number",
        value = guage_mean*10.1,
        domain = {'x': [0.5, 1], 'y': [0, 1]},
        title = {'text': title1},
        # delta = {'reference': delta},
        gauge = {'axis': {'range': [0, 1]}})
                    )
    gauge_fig_2 = go.Figure()
    gauge_fig_2.add_trace(
        go.Indicator(
        mode = "gauge+number",
        value = guage_mean *10,
        domain = {'x': [0,0.5 ], 'y': [0, 1]},
        title = {'text': title2},
        # delta = {'reference': delta},
        gauge = {'axis': {'range': [0, 1]}})
        )
    
    
    gauge_fig_1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )
    gauge_fig_2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )

    gauge_fig_1.update_layout(height=500, width=650)
    
    gauge_fig_2.update_layout(height=500, width=650)

    col1.plotly_chart(gauge_fig_1)

    col2.plotly_chart(gauge_fig_2)
    # df_website = get_website_data()
    # # df_website = df_website[["oneplus","date"]]
    # # st.write(df_website['date'].dt.month)
    # by_month = df_website[['oneplus', 'samsung']]
    # by_month = by_month.groupby([df_website['date'].dt.year]).sum()
    # st.dataframe(by_month)
    # st.write(by_month.index)
    # website_fig = by_month.plot(title = "Website Traffic By Year")
    # website_fig.update_layout({
    #     'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #     'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #     },
    #         )
    # website_fig.update_xaxes(title_text='Date')
    # website_fig.update_yaxes(title_text='Visits in Millions')
    # st.plotly_chart(website_fig)

    
    
    traffic_df = get_website_data()
    df_new = traffic_df.loc[(traffic_df['year'] >= min) & (traffic_df['year'] <= max)]
    traffic_fig = go.Figure()

    traffic_fig.add_trace(go.Scatter(x=df_new["Year_Month"], y= df_new["oneplus"], name='Oneplus',
                             line=dict( width=4, color='rgb(131, 90, 241)'), fill='tozeroy'))
    traffic_fig.add_trace(go.Scatter(x=df_new["Year_Month"], y= df_new["samsung"], name='Samsung',
                             line=dict( width=4, color='rgb(111, 231, 219)'), fill='tozeroy'))
    traffic_fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )
    traffic_fig.update_layout(title_text='Website Visitors', title_x=0.5, autosize=False,  height = 500)
    traffic_fig.update_xaxes(title_text='Date Range', showgrid=False)
    traffic_fig.update_yaxes(title_text='Visits in Millions',showgrid=False)
    col2.plotly_chart(traffic_fig)
    my_expander = col2.beta_expander("Expand For More Details about above chart", expanded=False)
    with my_expander:
        my_expander.write("### This Data is from Similarweb!")
        my_expander.markdown("Last **6 years** traffic data from various platforms")
        my_expander.dataframe(traffic_df)
    # col1.markdown("***")


    

    
    

    gsm_samsung_df = gsm_entities_viz(brand = "samsung", minYear = min, maxYear =max )
    gsm_oneplus_df = gsm_entities_viz(brand = "oneplus", minYear = min, maxYear =max )

    gsm_fig = go.Figure()

    gsm_fig.add_trace(go.Scatter(x=gsm_samsung_df["Year_Month"], y= gsm_samsung_df["NewSentiment"], name='Oneplus',
                             line=dict(color='lightblue', width=4), fill='tozeroy'))
    gsm_fig.add_trace(go.Scatter(x=gsm_oneplus_df["Year_Month"], y= gsm_oneplus_df["NewSentiment"], name='Samsung',
                             line=dict(color='blue', width=4), fill='tozeroy'))
    gsm_fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )
    gsm_fig.update_layout(title_text='Review Scores', title_x=0.5,autosize=False,  height = 500)
    gsm_fig.update_xaxes(title_text='Date Range', titlefont=dict(size=15),showgrid=False)
    gsm_fig.update_yaxes(title_text='Model Name', titlefont=dict(size=10), showgrid=False)
    col1.plotly_chart(gsm_fig)
    my_expander = col1.beta_expander("Expand For More Details about above chart", expanded=False)
    with my_expander:
        my_expander.write("### This Data is from GSM Arena!")
        my_expander.markdown("Over **75k** reviews has been scraped and performed a Sentiment Analysis")
        my_expander.dataframe(gsm_samsung_df)
    

    df_benchmark = get_benchmark()
    #OnePlus
    by_brand = df_benchmark.loc[df_benchmark['Brand'] == "OnePlus"]
    by_brand = by_brand[['Model', 'Score']]
    by_brand = by_brand.sort_values('Score')
    # st.write(by_brand)
    bench_fig = go.Figure(go.Bar(
            y=by_brand["Model"],
            x=by_brand["Score"],
            orientation='h'))
    bench_fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )
    bench_fig.update_layout(title_text='Mobile Performance Metrics', title_x=0.5, autosize=False,  height = 500)
    bench_fig.update_xaxes(title_text='Benchmark Score')
    bench_fig.update_yaxes(title_text='Model Name')
    col1.plotly_chart(bench_fig)
    # st.write(by_brand.plot())
    bench_ex = col1.beta_expander("Expand For More Details about Benchmark Data", expanded=False)
    with bench_ex:
        bench_ex.write("### This Data is Collected from [Geekbench5](https://browser.geekbench.com/) & [Antutu!](https://www.antutu.com/en/index.htm)")
        bench_ex.markdown("This Data is from the compilation of **1M+** benchmark socres tested by the users")
        bench_ex.dataframe(by_brand)

    #Samsung
    by_brand = df_benchmark.loc[df_benchmark['Brand'] == "Samsung"][0:16]
    by_brand = by_brand[['Model', 'Score']]
    by_brand = by_brand.sort_values('Score')
    # st.write(by_brand)
    bench_fig = go.Figure(go.Bar(
            y=by_brand["Model"],
            x=by_brand["Score"],
            orientation='h'))
    bench_fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        )
    bench_fig.update_layout(title_text='Mobile Performance Metrics', title_x=0.5, autosize=False,  height = 500)
    bench_fig.update_xaxes(title_text='Benchmark Score',showgrid=False)
    # bench_fig.update_yaxes(title_text='Model Name',showgrid=False)
    col2.plotly_chart(bench_fig)
    # st.write(by_brand.plot())
    bench_ex = col2.beta_expander("Expand For More Details about Benchmark Data", expanded=False)
    with bench_ex:
        bench_ex.write("### This Data is Collected from [Geekbench5](https://browser.geekbench.com/) & [Antutu!](https://www.antutu.com/en/index.htm)")
        bench_ex.markdown("This Data is from the compilation of **1M+** benchmark socres tested by the users")
        bench_ex.dataframe(by_brand)
    
    
    # col1.plotly_chart(gauge_fig)

#############################################Opinion Analysis#############################

def opinion_analysis():

    # st.sidebar.balloons()
    st.info("Our Work on **Opinion Analysis** is in progress, Thanks for visiting us 🙂")
    
    # opinion_date_range = st.sidebar.slider('Select the Date Range', 2016, 2020, (2018, 2020), 1)
    # min,max = opinion_date_range[0],opinion_date_range[1]
    #TOodo
    # df = get_gsm_entities()
    # brands = df["brand"].value_counts()
    # opinion_select_brand = st.sidebar.multiselect(
    # 'Select the Brand ',
    # ['Oneplus', 'Samsung'], default = ["OnePlus"]
    # )
    # models = df[(df["brand"] == opinion_select_brand)]["model"].unique()
    # st.write(models)
    # opinion_select_model = st.sidebar.multiselect(
    # 'Select the Brand ',
    #     models , default = ["s9"]
    # )

    # gsm_data_df = o_gsm_entities_viz(minYear = min, maxYear=max)


    # gsm_fig_opinion = go.Figure()

    # gsm_fig_opinion.add_trace(go.Scatter(x=gsm_data_df["Year_Month"], y= gsm_data_df["NewSentiment"], name='Oneplus',
    #                          line=dict(color='lightblue', width=4), fill='tozeroy'))
    # gsm_fig_opinion.update_layout({
    #     'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #     'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #     },
    #     )
    # st.plotly_chart(gsm_fig_opinion)


##########################################################################################

###############################Stats###########################

def stats_analysis():

    st.markdown("<h2 style='text-align: center; color: black;'><b>OneSight Statistical Cognition 🕵️<b></h1>", unsafe_allow_html=True)
    option = st.selectbox(
        'Select the Statistical Model',
        ('Welch-T Test (For Unequal Variances)', 'ANOVA'))
    if option ==  "Welch-T Test (For Unequal Variances)":

        with st.spinner("**Sit Back**, OneSight is reasoning 🧠 "):
            time.sleep(1)
      
        # stat_box = st.selectbox("Select the  Stat", options = [1,2])
        # st.write(stat_box)
        stats = Image.open("..//assets//stats.png")
        
        st.markdown("### **Welch-T Test (For Unequal Variances)**")
        st.markdown("### **Hypothesis:**  _A user review is more likely to have Negative information placed at the end than at the beginning of the review._ ")
        st.markdown("We are applying T-test for unequal variances here. To test this hypothesis, we have analyzed and calculated the sentiment of first half first and then second half of each review and then summing the products of coefficient and their respective weighted term frequency. Which is referred as μ1 and μ2 respectively.  ")
        
        st.markdown("Summary statistics of μ1, μ2 and the document sentiment μ are available below. ")
        st.markdown("**H0: μ1 <= μ2 (law of primacy applied to negative content)**")
        st.markdown("**H1: μ1>μ2(regency effect for negative content)**") 
        st.image(stats, width = 650)
        st.info("The values in the above table are computed from GSMArean Reviews (Considered 75k reviews)")
        st.markdown(" **Panel 1**: All Customer Reviews")
        st.markdown("p-value equals 1.00000, **( p(x≤T) = 0.500000 )**.")
        st.markdown("The test statistic T equals 0.00000, is in the 95% critical value accepted range: **[-1.9618 : 1.9618]**.")
        st.markdown("Since p-value > α, **H0 is accepted**. ")
        st.markdown(" **Panel 2**: Reviews for Positive Rating ")
        st.markdown("p-value equals **0.242301**, ( p(x≤T) = 0.121151 ). ")
        st.markdown("The test statistic T equals -1.180621, is in the 95% critical value accepted range: **[-1.9993 : 1.9993]**. ")
        st.markdown("Since p-value > α, **H0 is accepted**. ")
        st.markdown(" **Panel 3**: Reviews for Negative Rating ")
        st.markdown("p-value equals **1.00000**, ( p(x≤T) = 0.500000 ). ")
        st.markdown("The test statistic T equals **0.00000**, is in the 95% critical value accepted range: [-1.9618 : 1.9618]. ")
        st.markdown("Since p-value > α, **H0 is accepted**. ")
        st.markdown("**Based on above information, we accept our hypothesis regarding the presence of a primacy effect. **")

    elif option == 'ANOVA':

        with st.spinner("**Sit Back**, OneSight is reasoning 🧠 "):
            time.sleep(1)
        stats2 = Image.open("..//assets//stats2.png")
        st.markdown("### **ANOVA**")
        st.markdown("### **Hypothesis:**  _To infer if there is a significant difference between the reviews for OnePlus and Samsung based on their sentiment score. _ ")
        st.markdown("We are applying T-test for unequal variances here. To test this hypothesis, we have analyzed and calculated the sentiment of all reviews for OnePlus and Samsung, which is referred as μ1 and μ2 respectively. ")
        st.markdown("Summary statistics of μ1, μ2 and the document sentiment μ are available below. We then present the statistics for reviews that are filtered for a positive (Section II) or negative (Section III) standard only. We then test the null hypotheses. ")
        st.markdown("**Null Hypothesis** - H0: μ1 – μ2 = 0 ")
        st.markdown("**Alternate Hypothesis** - Ha: μ1 – μ2 ≠ 0 ")
        st.image(stats2, width = 650)
        st.info("The values in the above table are computed from GSMArean Reviews (Considered 75k reviews)")
        st.markdown(" **Panel 1 _(All Reviews)_**: H0 hypothesis")
        st.markdown("Since p-value < α, H0 is rejected")
        st.markdown("The average of **OnePlus reviews** is considered to be **not equal to** the average of the **Samsung reviews**. In other words, the difference between the average of two is big enough to be statistically significant. ")
        st.markdown("p-value equals **0.0353695** ")
        st.markdown("The Statstics : The test statistic T equals **2.109186** ")
        st.markdown("***")
        st.markdown(" **Panel 2 _(Positive Reviews)_**: H0 hypothesis")
        st.markdown("Since p-value > α, H0 is accepted")
        st.markdown("The average of OnePlus reviews is considered to be equal to the average. of the Samsung reviews. In other words, the difference between the average of two is not big enough to be statistically significant.")
        st.markdown("p-value equals **0.446919** ")
        st.markdown("The Statstics : The test statistic T equals **0.772597** ")
        st.markdown("***")
        st.markdown(" **Panel 3 _(Negative Reviews)_**: H0 hypothesis")
        st.markdown("Since p-value > α, H0 is accepted")
        st.markdown("The average of **OnePlus reviews** is considered to be **equal to** the average of the **Samsung reviews**. In other words, the difference between the average of two is not big enough to be statistically significant.")
        st.markdown("p-value equals **0.333893** ")
        st.markdown("The Statstics : The test statistic T equals **0.971027** ")
        st.markdown("_We have seem from the above results that there is a significant difference between the overall customer sentiment for Samsung and OnePlus as seen in panel 1_")
        st.markdown("_However, we found that the both Brands have a similar sentiments for Positively and Negatively rated products as seen in panel 2 and 3._")



############################################################# Chatbot ####################3
def chatbot():
    import streamlit.components.v1 as components

# bootstrap 4 collapse example
    # components.html(
    # """
    # <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
    # <df-messenger
    # intent="WELCOME"
    # chat-title="test-demo"
    # agent-id="f249e6e0-c3bc-470f-a4e0-9445c9bd7c20"
    # language-code="en">
    # </df-messenger>
    # """,
    #     height=700,
    #     width = 800
    # )
    components.html(
        """
        <iframe width="1050" height="430" allow="microphone;" src="https://console.dialogflow.com/api-client/demo/embedded/f249e6e0-c3bc-470f-a4e0-9445c9bd7c20"></iframe>
        """, height = 430, width = 1050
    )

    
###############################################################
user_preference_options = [ "Competitive Analysis🔍", "OneSight Statistical Cognition 🕵️", "Reporting Live Data 🔴📡 ", "Opinion Analysis", "AI Assistant 🧠 💬 👋 "]
user_preference = st.sidebar.radio(label="Want to know?", options=user_preference_options, index=0)

if user_preference == user_preference_options[3]:
    opinion_analysis()
elif user_preference == user_preference_options[2]:
    twitter_analyser()
elif user_preference == user_preference_options[0]:
    competitive_analysis()
elif user_preference == user_preference_options[1]:
    stats_analysis()
elif user_preference == user_preference_options[4]:
    chatbot()

# add_selectbox = st.sidebar.multiselect(
#     'Select the company ',
#     ['OnePlus', 'Samsung'], default = "OnePlus"
# )

# selected_date_range = st.sidebar.slider('Select the Date Range', 2016, 2020, (2018, 2020), 1)
#Access as below
#selected_date_range[0]

# st.sidebar.radio()
# st.info("**Sit Back**, OneSight is reasoning 🧠 ")

st.sidebar.title("About")
st.sidebar.info(
        """
        This project is created & maintained by **One Sight Team**. You can learn more about us at
        [our documentation](https://github.com/vasudevmaduri/OnePlus-Analysis/tree/dev).

        **Our Team**:
        - [Sushmita Sahu](https://www.linkedin.com/in/sushmita-sahu-764b1865/)
        - [Shoumitra Biswas](https://www.linkedin.com/in/shoumi786/)
        - [Mahima Sharma](https://www.linkedin.com/in/mahima-sharma/)
        - [Yash Srivastava](https://www.linkedin.com/in/yash-srivastava-2b266515a/)
        - [Vasudev Maduri](https://www.linkedin.com/in/vasudevmaduri/)
"""
    )
# hide_streamlit_style = """
#             <style>
#             # MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.info(" Made with ❤️ In India ")





