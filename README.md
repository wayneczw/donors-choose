# DonorsChoose

## Raw Data
DonorsChoose.org Application Screening
data type: csv size: 195MB binary classification problem https://www.kaggle.com/c/donorschoose-application-screening

## Environment Setup

	pip install -r requirements.txt

	mkdir weights

	mkdir data

	mkdir results

#### Large File Handling

	git lfs install

	git lfs track "*.csv"

	git add .gitattributes


#### Download fastText Word Vector

[Download](https://fasttext.cc/docs/en/english-vectors.html) fastText Word Vector - crawl-300d-2M.vec.zip. Unzip the downloaded file, and save into [data](./data/)

## Preprocess
As running the preprocessing steps requires a lot of time, we therefore save all the processed data in this [GDrive](https://drive.google.com/open?id=1ACuaxLLt55OS-Hj-0GNxVPfhpILqul-0)

To rerun the preprocessing steps, follow the following sections.

#### Prepare Config YAML
In [benchmark.yaml](benchmark.yaml), edit all the necessary input path. For example, *word_path: data/crawl-300d-2M.vec* is required to be directed to the dir where the fastText word vector is at.

#### Preprocess Raw Data
In [benchmark.yaml](benchmark.yaml), make sure the following keys are modified to point to the rightful unprocessed raw train/test/resources csv:
- train: data/train.csv
- test: data/test.csv
- resources: data/resources.csv
 
Run the following to preprocess the data (this process may take a while as there are many feature engineerings involved):

	python -m preprocess

After running [preprocess.py](preprocess.py), the following processed csv will be output:
- data/train_processed.csv
- data/test_processed.csv

#### Save Universal Sentence Encoder (USE) vectors
We run USE over all text columns once, and save the encoded vectors via the following command:

	python -m save_use_text

After running [save_use_text.py](save_use_text.py), the following joblib files will be output:
- data/train_use.joblib
- data/test_use.joblib

## Model Training with Different Experimental Setup
This section briefly describes how to modify the config yaml file [benchmark.yaml](benchmark.yaml), so as to run different experimental settings. To be specific, we present 2 environmental set ups:
- benchmark model setup
- multi-channel USE model

#### Prepare Config YAML
In [benchmark.yaml](benchmark.yaml), edit all the necessary input path. For example, *word_path: data/crawl-300d-2M.vec* is required to be directed to the dir where the fastText word vector is at.

Also, ensure the following directories are modified:
- train: data/train_processed.csv
- test: data/test_processed.csv
- train_use: data/train_use.joblib
- test_use: data/test_use.joblib

#### Benchmark Model
For benchmark model, only set those labeled with **#benchmark** in [benchmark.yaml](benchmark.yaml) as **true**.

And also ensure the following keys to output submission csv files:
- train_output_csv: benchmark_train.csv #to be changed for each experiment
- test_output_csv: benchmark_train_test.csv #to be changed for each experiment

*train_output_csv is for ensemble learning later on*

*test_output_csv is for kaggle submission*


Run the following to train the model:

	python -m benchmark


The model trained under this experimental set up correspond to the model **Model: Benchmark** and **Features: Benchmark** stated in the report, with Kaggle public score of 0.70631.

#### Multi-channel USE model
For multi-channel USE model, ensure those labeled with **#benchmark** in [benchmark.yaml](benchmark.yaml) as **true**. But also change the embedding section to the following:

	# Text - only set one of them to be True
	embedding:
	  tfidf: False #benchmark
	  use: True
	  word_vector: False
	  word_path: data/crawl-300d-2M.vec


And also ensure the following keys to output submission csv files:
- train_output_csv: use_train.csv #to be changed for each experiment
- test_output_csv: use_train_test.csv #to be changed for each experiment

*train_output_csv is for ensemble learning later on*

*test_output_csv is for kaggle submission*

Run the following to train the model:

	python -m benchmark

The model trained under this experimental set up correspond to the model **Model: USE MC** and **Features: Benchmark** stated in the report, with Kaggle public score of 0.75960.


## Ensemble Learning
Before writing the [ensemble_learning.py](ensemble_learning.py) script, make sure that all the csv files which contain the predication probabilities for both the training set and test set for the 7 base models are present inside a folder called [results](./results).

Specially, we will need the following 14 csv files in the folder [results](./results):
- 'rfc_best_test.csv',
- 'lgbm_best_test.csv',
- 'word_vector_best_test.csv',
- 'xgboost_best_test.csv',
- 'ftrl_best_test.csv',
- 'nn_use_best_test.csv',
- 'dpcnn_s2.3_countenco_test.csv'
- 'rfc_best_train.csv',
- 'lgbm_best_train.csv',
- 'word_vector_train.csv',
- 'xgboost_best_train.csv',
- 'ftrl_best_train.csv',
- 'nn_use_best_train.csv',
- 'dpcnn_s2.3_countenco_train.csv'

Run the following to ensemble the base models:

	python -m ensemble_learning

After running is completed, the final result output will be called "ensemble_output.csv" and it is stored in [results](./results).

