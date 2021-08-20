import pandas as pd
import numpy as np
from category_encoders.binary import BinaryEncoder


TRAIN_FILE = "./adult.data"
TEST_FILE = "./adult.test"

INDEX_TO_CHANGE = [1,3,5,6,7,8,13]



def read_data():

	train_data = np.array(pd.read_csv(TRAIN_FILE, header=None))

	test_data = np.array(pd.read_csv(TEST_FILE, header=None))
	
	data = np.concatenate((train_data, test_data), axis=0)

	data = data[:, :-1]

	encoder = BinaryEncoder(verbose=1, cols=INDEX_TO_CHANGE)
	encoder.fit(data)

	final_data = encoder.fit_transform(data)

	COLUMNS_TO_DROP = []
	for col in final_data.columns:
		if np.sum(np.array(final_data[col])) == 0:
			COLUMNS_TO_DROP.append(col)

	final_data = final_data.drop(columns=COLUMNS_TO_DROP)

	final_data[9][final_data[9] == " Male"] =  1
	final_data[9][final_data[9] == " Female"] =  0
	final_data = final_data.rename(columns={'9': 'GENDER'})
	print(final_data.columns)
	final_data.to_csv('adult.csv')


if __name__ == "__main__":

	adult_data = read_data()