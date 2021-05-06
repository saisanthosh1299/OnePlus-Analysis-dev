import pandas as pd
# gs://foundationalproject1/Cell_Phones_and_Accessories.json
df = pd.read_json ('Cell_Phones_and_Accessories.json')
export_csv = df.to_csv (r'New_Products.csv', index = None, header=True)