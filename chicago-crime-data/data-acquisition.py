'''
Download reported crime data from the Chicago open data portal for 2017 and
2018. Generate summary statistics for the crime reports data.
'''

import pandas as pd
from sodapy import Socrata

# resource for code snippets for getting started with sodapy:
# https://dev.socrata.com/foundry/data.cityofchicago.org/6zsd-86xi

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofchicago.org", None)

# Results returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
# Number of expected results: 534373
results = client.get("6zsd-86xi", where="year=2017 OR year=2018", limit=550000)

results_df = pd.DataFrame.from_records(results)
