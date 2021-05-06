import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = 'https://browser.geekbench.com/android-benchmarks'
page = requests.get(URL)

data=[]
soup = BeautifulSoup(page.content, 'html.parser')

job_elems = soup.find("table",{'id':'android'})

table_body = job_elems.find('tbody')

rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele]) # Get rid of empty values


df = pd.DataFrame(data)

df.to_csv("benchmark.csv")

#Table below

# <tr>
# <td class="name">
# <div class="device-icon samsung"></div>
# <a href="/android_devices/samsung-galaxy-note10-qualcomm-snapdragon-855-1-8-ghz">
# Samsung Galaxy Note10+
# </a>
# <div class="description">
# Qualcomm Snapdragon 855 @ 1.8 GHz
# </div>
# </td>
# <td class="score">
# 2478
# </td>
# <td class="graph">
# <div class="benchmark-bar" style="width:77%">
#  
# </div>
# </td>
# </tr>