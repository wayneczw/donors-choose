# DonorsChoose

## Data
DonorsChoose.org Application Screening
data type: csv size: 195MB binary classification problem https://www.kaggle.com/c/donorschoose-application-screening

## Data fields
test.csv and train.csv:

id - unique id of the project application
teacher_id - id of the teacher submitting the application
teacher_prefix - title of the teacher's name (Ms., Mr., etc.)
school_state - US state of the teacher's school
project_submitted_datetime - application submission timestamp
project_grade_category - school grade levels (PreK-2, 3-5, 6-8, and 9-12)
project_subject_categories - category of the project (e.g., "Music & The Arts")
project_subject_subcategories - sub-category of the project (e.g., "Visual Arts")
project_title - title of the project
project_essay_1 - first essay*
project_essay_2 - second essay*
project_essay_3 - third essay*
project_essay_4 - fourth essay*
project_resource_summary - summary of the resources needed for the project
teacher_number_of_previously_posted_projects - number of previously posted applications by the submitting teacher
project_is_approved - whether DonorsChoose proposal was accepted (0="rejected", 1="accepted"); train.csv only
* Note: Prior to May 17, 2016, the prompts for the essays were as follows:

project_essay_1: "Introduce us to your classroom"
project_essay_2: "Tell us more about your students"
project_essay_3: "Describe how your students will use the materials you're requesting"
project_essay_4: "Close by sharing why your project will make a difference"
Starting on May 17, 2016, the number of essays was reduced from 4 to 2, and the prompts for the first 2 essays were changed to the following:

project_essay_1: "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."
project_essay_2: "About your project: How will these materials make a difference in your students' learning and improve their school lives?"
For all projects with project_submitted_datetime of 2016-05-17 and later, the values of project_essay_3 and project_essay_4 will be NaN.

resources.csv:

Proposals also include resources requested. Each project may include multiple requested resources. Each row in resources.csv corresponds to a resource, so multiple rows may tie to the same project by id.

id - unique id of the project application; joins with test.csv. and train.csv on id
description - description of the resource requested
quantity - quantity of resource requested
price - price of resource requested

## Environment Setup

	pip install -r requirements.txt

#### Download fastText Word Vector

[Download](https://fasttext.cc/docs/en/english-vectors.html) fastText Word Vector - crawl-300d-2M.vec.zip.

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

## Experiment
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
