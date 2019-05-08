'''
Jessica Langston
CAPP 30254 HW 3 - Improving the Pipeline

Library of functions for use in machine learning pipeline
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy


def load_csv_data(filename, dtype=None):
	'''
	Takes a CSV file and loads it into a pandas dataframe

	Input:
		filename (str) - CSV filename
		dtypes (dict) - optional dictionary mapping column names to data types
	Returns:
		pandas dataframe
	'''
	return pd.read_csv(filename, dtype=dtype)


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
	print(df[col].value_counts().plot(kind="bar"))


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


def cols_to_datetime(df, cols):
	'''
	Convert one or more columns of a dataframe to datetime type.

	Input:
		df (pandas dataframe)
		cols (list of strings) - columns to convert
	Returns:
		nothing - modifies dataframe in place
	'''
	for col in cols:
		df[col] = pd.to_datetime(df[col])


def fill_na_with(df, method, cols=None):
	'''
	Find columns with null values and fill those null values with the mean,
	median, or mode value of the column.

	Input:
		df (pandas dataframe)
		method (string) - measure of center to use to fill null values
		cols (list of strings) - columns to check for null values. If None,
			fill nulls in all columns
	Returns:
		nothing - modifies dataframe in place
	'''
	if not cols:
		cols = []
		for col, has_null in df.isnull().any().iteritems():
			if has_null:
				cols.append(col)

	fill_values = {}
	for col in cols:
		if method == "mean":
			val = df[col].mean()
		if method == "median":
			val = df[col].median()
		if method == "mode":
			val = df[col].mode()[0]

		num_nulls = df[col].isnull().sum()
		fill_values[col] = val
		print("Replacing {} nulls in column {} with {} value {}".format( \
			num_nulls, col, method, val))

	df.fillna(value=fill_values, inplace=True)


def make_discrete(df, col, bins, labels):
	'''
	Transform the given continuous variable column into a discrete variable.

	Inputs:
		df (pandas dataframe)
		col (string) - name of column to transform
		bins (list of ints) - boundaries for new discrete variable
		labels (list of strings) - labels for bins
	Returns:
		nothing - modifies dataframe in place
	'''
	df[col] = pd.cut( \
		df[col], bins=bins, labels=labels, include_lowest=True, right=False)


def make_dummy_from_continuous(df, col, cutoff, gt_cutoff=True, new_col=None):
	'''
	Make dummy variable column from the given numerical variable column.
	Dummy variable will be coded as 0 when the original variable takes a value
	below the cutoff and 1 when the variable takes on a value of the given
	cutoff or more (or vice versa, if gt_cutoff is set to False).

	Inputs:
		df (pandas dataframe)
		col (string) - name of column to make dummy from
		cutoff (int or float) - value above which dummy var should equal 1
		gt_cutoff (boolean) - if True, the dummy variable should take a
			value of 1 if the value is greater than the cutoff. Pass False
			when the dummy should equal 1 if the value is less than cutoff
		new_col (string) - optional argument to specify name of new column,
			otherwise defaults to adding "_dummy" suffix to original col name
	Returns:
		nothing - modifies dataframe in place
	'''
	if not new_col:
		new_col = col + "_dummy"
	df[new_col] = 0
	if gt_cutoff:
		df.loc[df[col] >= cutoff, new_col] = 1
	else:
		df.loc[df[col] <= cutoff, new_col] = 1


def make_dummy_from_categorical(df, col, map_to_dummy, new_col=None):
	'''
	Make dummy variable column from the given categorical variable column.
	Dummy variable will be coded as 1 for all values in the list argument
	map_to_dummy and 0 otherwise.

	Inputs:
		df (pandas dataframe)
		col (string) - name of column to make dummy from
		map_to_dummy (list) - values that should map to 1 in the dummy column
		new_col (string) - optional argument to specify name of new column,
			otherwise defaults to adding "_dummy" suffix to original col name
	Returns:
		nothing - modifies dataframe in place
	'''
	if not new_col:
		new_col = col + "_dummy"
	df[new_col] = 0
	for val in map_to_dummy:
		df.loc[df[col] == val, new_col] = 1


def rename_values(df, col, names):
	'''
	Given a dictionary of mappings of current names/values to new names/values,
	update the values in the specified column. For example, if in column "A"
	we want to change all instances of the value "Yes" into "Y", the names dict
	should contain a key-value pair {"Yes": "Y"}.

	Inputs:
		df (pandas dataframe)
		col (string) - name of column
		names (dict) - maps current names to desired names for values in column
	Returns:
		nothing - modifies dataframe in place
	'''
	for old_name, new_name in names.items():
		df.loc[df[col] == old_name, col] = new_name


def split_data(df, selected_features, label, test_size):
	'''
	Using the columns specified as features, split the data to obtain
	X and y training and testing sets.

	Inputs:
		df (pandas dataframe)
		selected_features (list of strings) - names of columns to be used as
			features
		label (string) - name of column to be used as the label
		test_size - proportion of data to use as testing data
	Returns:
		array containing x_train, x_test, y_train, and y_test dataframes
	'''
	X = df[selected_features]
	y = df[label]

	return train_test_split(X, y, test_size=test_size)


def fit_decision_tree(x_train, y_train):
	'''
	Use the training X and y data to fit the decision tree.

	Inputs:
		x_train (pandas dataframe) - features training data
		y_train (pandas dataframe) - label training data
	Outputs:
		dec_tree (DecisionTreeClassifier object)
	'''
	dec_tree = DecisionTreeClassifier()
	dec_tree.fit(x_train, y_train)

	return dec_tree


def fit_knn_classifier(x_train, y_train, n_neighbors):
	'''
	Use the training X and y data to fit a k-nearest neighbors classification
	model.

	Inputs:
		x_train (pandas dataframe) - features training data
		y_train (pandas dataframe) - label training data
		n_neighbors (int) - number of neighbors to use in the model
	Outputs:
		knn (KNeighborsClassifier object)
	'''
	knn = KNeighborsClassifier(n_neighbors, metric="minkowski")
	knn.fit(x_train, y_train)

	return knn


def evaluate_model(predicted_scores, y_test, threshold):
	'''
	Evaluate the accuracy of the trained model.

	Inputs:
		predicted_scores (numpy array) - a list of probabilities, one for each
			test data point, that the data point belongs to class 1
		y_test (pandas dataframe) - label testing data
		threshold (float) - if predicted score is above this threshold, 		consider it to be an accurate prediction
	Outputs:
		test_acc (float)

	Source: Code borrowed/modified from ML for Public Policy lab 2
	'''
	# predicted_scores_test = trained_model.predict_proba(x_test)[:,1]
	# plt.hist(predicted_scores_test)

	calc_threshold = lambda x, y: 0 if x < y else 1
	predictions = np.array( \
		[calc_threshold(score, threshold) for score in predicted_scores])
	
	test_acc = accuracy(predictions, y_test)

	print("Model accuracy is {}".format(test_acc))
