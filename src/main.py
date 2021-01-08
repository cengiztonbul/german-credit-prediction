import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	df = pd.read_csv('data/german-credit.data')
	df = shuffle(df)
	
	categorized = pd.get_dummies(df, columns=['checking_account_status', 
	'credit_history', 'purpose', 'savings_account_status', 'employment_years',  
	'personal_status', 'guarantors', 'property', 'installment_plans', 'housing', 
	'job', 'telephone', 'foreign_worker'])

	train, test = train_test_split(categorized, test_size=0.2)

	attributes = train.loc[:, train.columns != 'credit_status']
	credit_status = train.loc[:,['credit_status']]

	attributes_test = test.loc[:, train.columns != 'credit_status']
	credit_status_test = test.loc[:,['credit_status']]

	from sklearn.linear_model import LogisticRegression
	lr = LogisticRegression(n_jobs=-1, max_iter=20000)
	lr.fit(attributes, credit_status.values.ravel())
	
	pred = lr.predict(attributes_test)
	score = accuracy_score(credit_status_test.values.ravel(), pred.round())
	print(score)