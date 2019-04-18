'''
Library of functions for use in machine learning pipeline
'''

import pandas as pd


def load_csv_data(filename):
	'''
	TODO
	'''
	return pd.read_csv(filename)


def show_data_summary(df, col=None):
	'''
	TODO
	'''
	print(df.head())
	print(df.dtypes)
	print(df.describe())
	df["colname"].describe()
	print(df.corr(method="pearson"))

	# remember: just do what will be helpful for this data
	# will modify for other types of data later

	# also: implemen for this data in another file
	# this module should be reusable


def show_histogram(df):
	'''
	TODO
	'''
	# can change column name
	col = "Age Group"
	df[col].value_counts().plot(kind="bar")
	# try plt.hist()
	# plt.hist(df["Age Group"])

	# try histogram splitting on outcome variable


def show_correlation(df, cols=None):
	'''
	TODO
	'''
	df.corr(method="pearson")
	df["PersonID"].corr(df["SeriousDlqin2yrs"], method="pearson")


def detect_outliers(df):
	'''
	TODO

	https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
	'''
	pass
