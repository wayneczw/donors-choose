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


## Benchmark
Run
	
	pip install -r requirements.txt

There are 2 scripts involved:
1. benchmark.yaml
* change the param settings in this script
* change train/test data path in this script
* change output path in this script
2. benchmark.py
* the script that runs the training
* command:

	python -m benchmark

#### Work distribution
See benchmark.yaml's comments for your parts to run.
