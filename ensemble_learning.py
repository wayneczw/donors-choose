from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

print("Doing Emsemble Learning")

# Edit 'mode' to use dif Ensembling Methods
mode = 0 # 0 for LGBM Ensemble, 1 for Simple Probabilities Average Ensemble, 2 for Weighted Average Probabilities Ensemble



test_model_name_list = ['rfc_best_test.csv','lgbm_best_test.csv', 'word_vector_best_test.csv', 'xgboost_best_test.csv', 'ftrl_best_test.csv', 'nn_use_best_test.csv', 'dpcnn_s2.3_countenco_test.csv']
score_list = [0.76208,0.80257,0.77927,0.74012,0.78022, 0.79242, 0.81640]
predict_test_list = []
for filename in test_model_name_list:
    filename = 'data/' + filename
    model_result = pd.read_csv(filename)
    predict_test_list.append(model_result)



if (mode == 0):
	# Start of LGBM Ensemble
	import lightgbm as lgb
	from sklearn.model_selection import train_test_split

	df = pd.read_csv('data/train.csv')
	df.sort_values(by='id', inplace=True)
	train_target = df['project_is_approved'].values

	train_model_name_list = ['rfc_best_train.csv', 'lgbm_best_train.csv', 'word_vector_train.csv', 'xgboost_best_train.csv', 'ftrl_best_train.csv', 'nn_use_best_train.csv', 'dpcnn_s2.3_countenco_train.csv' ]
	predict_train_list = []
	for filename in train_model_name_list:
	    filename = 'data/' + filename
	    model_result = pd.read_csv(filename)
	    model_result.sort_values(by='id', inplace=True)
	    predict_train_list.append(model_result)

	scaler = MinMaxScaler()

	train_results = np.hstack([scaler.fit_transform(item['project_is_approved'].values.reshape((item.shape[0], 1))) for item in predict_train_list])
	test_results = np.hstack([scaler.fit_transform(item['project_is_approved'].values.reshape((item.shape[0], 1))) for item in predict_test_list])
	train_features = np.hstack([train_results])
	test_features = np.hstack([test_results])

	lgb_model = lgb.LGBMClassifier(  n_jobs=4,
	                                 max_depth=4,
	                                 metric="auc",
	                                 n_estimators=400,
	                                 num_leaves=10,
	                                 boosting_type="gbdt",
	                                 learning_rate=0.01,
	                                 feature_fraction=0.45,
	                                 colsample_bytree=0.45,
	                                 bagging_fraction=0.4,
	                                 bagging_freq=5,
	                                 reg_lambda=0.2)
	X_train, X_val, y_train, y_val = train_test_split(train_features, train_target, train_size=0.8, random_state=233)
	lgb_model.fit(X=X_train, y=y_train,
	              eval_set=[(X_val, y_val)],
	              verbose=False)
	final_predict = lgb_model.predict_proba(test_features)[:,1]
	# End of LGBM Ensemble



elif (mode == 1):
	score_list = [1,1,1,1,1,1,1] # for simple average ensemble
	# Start of Simple Average - Ensemble

	sum = 0 
	score_list = [((i*10)**75)/10000 for i in score_list] # increasing the difference between elements in the score_list
	series_final = predict_test_list[0]['project_is_approved'].mul(score_list[0])
	sum = sum + score_list[0]
	count = 0
	for prob_result in predict_test_list:
	    if count == 0: # skip the first one since we already added it in
	        count = count + 1
	        continue
	    series_new = prob_result['project_is_approved']
	    series_final = series_final.add(series_new.mul(score_list[count]))
	    sum = sum + score_list[count]
	    count = count + 1
	series_final = series_final.div(sum)
	final_predict = series_final.values
	# End of Simple Average - Ensemble

else:

	# Start of Weighted Average - Ensemble
	# Weighted Average based on public score of each model
	sum = 0 
	score_list = [((i*10)**75)/10000 for i in score_list] # increasing the difference between elements in the score_list
	series_final = predict_test_list[0]['project_is_approved'].mul(score_list[0])
	sum = sum + score_list[0]
	count = 0
	for prob_result in predict_test_list:
	    if count == 0: # skip the first one since we already added it in
	        count = count + 1
	        continue
	    series_new = prob_result['project_is_approved']
	    series_final = series_final.add(series_new.mul(score_list[count]))
	    sum = sum + score_list[count]
	    count = count + 1
	series_final = series_final.div(sum)
	final_predict = series_final.values
	# End of Weighted Average - Ensemble




print("Emsemble Learning Done")
sample_df = pd.read_csv('data/sample_submission.csv')
sample_df['project_is_approved'] = final_predict
sample_df.to_csv('data/ensemble_output.csv', index=False)
