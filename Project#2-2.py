from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

def sort_dataset(dataset_df):
	#TODO: Implement this function
  sorted_df = dataset_df.sort_values(by='year', ascending=True)
  return sorted_df

def split_dataset(dataset_df):

    train_df = dataset_df.iloc[:1718]

    test_df = dataset_df.iloc[1718:]

    X_train = train_df.drop(columns="salary", axis=1)
    y_train = train_df["salary"] * 0.001

    X_test = test_df.drop(columns="salary", axis=1)
    y_test = test_df["salary"] * 0.001

    return X_train, X_test, y_train, y_test



def extract_numerical_cols(dataset_df):
    # Extracting only numerical columns
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']

    # Filtering numerical columns from the dataset
    numerical_df = dataset_df[numerical_cols]

    return numerical_df


def train_predict_decision_tree(X_train, Y_train, X_test):
    #TODO: Implement this function
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)
    dt_predictions = dt_model.predict(X_test)
    return dt_predictions

def train_predict_random_forest(X_train, Y_train, X_test):
	  #TODO: Implement this functionl
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)
    return rf_predictions

def train_predict_svm(X_train, Y_train, X_test):
    # TODO: Implement this function
    svr_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svr_pipe.fit(X_train, Y_train)
    svm_predictions = svr_pipe.predict(X_test)
    return svm_predictions

def calculate_RMSE(labels, predictions):
	ret = np.sqrt(np.mean((predictions-labels)**2))
	return ret

	#TODO: Implement this function

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

	sorted_df = sort_dataset(data_df)
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)

	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
