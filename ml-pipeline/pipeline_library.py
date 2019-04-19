'''
Library of functions for use in machine learning pipeline
'''

import pandas as pd


def load_csv_data(filename):
	'''
	Takes a CSV file and loads it into a pandas dataframe

	Input:
		filename (str) - CSV filename
	Returns:
		pandas dataframe
	'''
	return pd.read_csv(filename)


def show_data_summary(df):
	'''
	Prints a summary of the data in a pandas dataframe: head (first five rows),
	column names and corresponding data types, and descriptive statistics.

	Input:
		df (pandas dataframe)
	Returns:
		nothing
	'''
	print(df.head())
	print("")
	print(df.dtypes)
	print("")
	print(df.describe())


def show_label_detail(df, label):
	'''
	Show detail about the label, i.e. the outcome variable of interest. This
	function assumes the label is a binary variable coded as 0 for
	"no"/"false" and 1 for "yes"/"true"

	Input:
		df (pandas dataframe)
		label (str) - name of the column being used as the ML model label
			(i.e. outcome variable)
	'''
	print("Label value counts")
	print(df[label].value_counts())
	print("")
	print("Proportion of 'yes' observations")
	print(df[label].sum() / df.shape[0])


def show_histogram(df, col):
	'''
	Show a histogram of the data in the specified dataframe column

	Inputs:
		df (pandas dataframe)
		col (str) - column name
	'''
	df[col].value_counts().plot(kind="bar")


def show_correlation(df, cols=None):
	'''
	Show the correlation between all combinations of columns, or, if the
	optional parameter cols is passed in, show correlation between the
	specified columns

	Input:
		df (pandas dataframe)
		cols (pair)
	Returns:
		pandas dataframe of numpy float
	'''
	if cols:
		col1, col2 = cols
		return df[col1].corr(df[col2], method="pearson")

	return df.corr(method="pearson")


def preprocess_data(df):
	'''
	Find columns with null values and fill those null values with the mean
	value of the column.

	Input:
		df (pandas dataframe)
	Returns:
		nothing - modifies dataframe in place
	'''
	fill_values = {}
	for col, has_null in df.isnull().any().iteritems():
		if has_null:
			mean = df[col].mean()
			num_nulls = df[col].isnull().sum()
			fill_values[col] = mean
			print("Replacing {} nulls in column {} with mean value {}".format(num_nulls, col, mean))

	df.fillna(value=fill_values, inplace=True)



# FEATURES
# label - "SeriousDlqin2yrs"
# not sure about - PersonID, zipcode
# features - rest of the columns