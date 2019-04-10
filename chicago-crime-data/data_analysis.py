'''
- Download reported crime data from the Chicago open data portal for 2017 and
2018. 
- Generate summary statistics for the crime reports data.
'''

import pandas as pd
from sodapy import Socrata


def get_crime_data(years, num_results):
	'''
	Download Chicago crime data for indicated years with indicated maximum
	number of results and load into pandas dataframe.

	Input:
		years (list of int)
		num_results (int)
	Output:
		pandas dataframe
	'''
	# Resource for code snippets for getting started with sodapy:
	# https://dev.socrata.com/foundry/data.cityofchicago.org/6zsd-86xi

	# Unauthenticated client only works with public data sets. Note 'None'
	# in place of application token, and no username or password:
	client = Socrata("data.cityofchicago.org", None)

	where = ["year=" + str(year) for year in years]
	where_str = " OR ".join(where)

	# Results returned as JSON from API / converted to Python list of
	# dictionaries by sodapy.
	results = client.get("6zsd-86xi", where=where_str, limit=num_results)

	# load into dataframe and clean
	crime_df = pd.DataFrame.from_records(results)
	crime_df["date"] = pd.to_datetime(crime_df["date"])
	crime_df["month"] = crime_df["date"].dt.month

	return crime_df


def generate_summ_stats(crime_df):
	'''
	Given a dataframe of crime data, generate summary statistics.

	Input:
		crime_df (pandas dataframe)

	Output:
		none (side effect: print statements)
	'''

	# Number of crimes of each type
	type_counts = crime_df["primary_type"].value_counts()
	print(type_counts)

	# How the number of crimes of each type change over time
	gbtmy = crime_df.groupby(["primary_type", "month", "year"]).size()
	print(gbtmy.unstack().unstack())

	# How the number of crimes are different by neighborhood
	gbtc = crime_df.groupby( \
		["primary_type", "community_area"]).size().unstack()
	print(gbtc)
