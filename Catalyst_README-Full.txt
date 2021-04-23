Basic workflow
1. The recommended_courses block is added to a course
2. When a user visits the course page, the block will display some courses.


How it works:
1. See the section architecture heading in local_recommender


Machine Learning Library
# Model selected.
* The model we are using is JCA.
* This model does not care about the timestamp and only cares about whether a value is true or false; false is implicit.
* The reason this model was selected is for simplicity.
* Subject to testing, we can change the model if needed.


# Configuration and changing the model.
* It is easy enough to change the model being used in NeuRec.properties. Simply change the string that specifies which model is being used.
* The values in NeuRec.properties take precedence over the properties found in conf/{model}.properties
* In NeuRec.properties you can specify the format of the dataset. It’s currently set to UIR which means the first column of the dataset is user, the second is item, and the third is rating. If you specify UIRT then the fourth column will be timestamp.


# Overview.
* This is the brain of the machine learning. It ingests the data and writes out the results for each user into a csv.
* It currently writes out the top 20 ratings for each user.
* moodle-local_recommender (mlr) will handle removing courses that the user has already completed.


# Set up.
* dataset/ should be mapped to a shared directory.
* mlr should have access to this directory.
* When mlr runs it will generate a dataset and place it in this dir.
    * This dataset is our training data.


# Dependencies.
* At time of writing, this library works with python3.6 but not 3.8 
* python3.6-dev
* python3.6
* python3.6-venv
* build-essential
* pip3.6 dependencies:
    * guppy3
    * cython
    * setuptools


# Other info.
* The training takes a long time to run. It's been configured to run 100 epochs (iterations) and each epoch takes ~1000secs to run.
* This means to complete all 100 epochs it will take ~28 hours with a sample size of 170k records.
* As there becomes more completion records, the sample size will also increase, meaning that epochs will take longer.
* The trained model can be found in dataset/_tmp_{datasetname}
    * You shouldn't need to do anything with it, but this is where it is nonetheless.
* Due to the size of the dataset, it needs a large amount of memory to instantiate the model object.
    * This is because in order to create the model object it ingests the entire dataset and maps it into a 2D array using numpy.eye(), twice.


# VM setup steps.
* These are the steps I used in order to set up a VM and get it running.
1. sudo add-apt-repository ppa:deadsnakes/ppa
2. sudo apt-get update
3. sudo apt install python3.6-dev python3.6 python3.6-venv build-essential
4. python3.6 -m venv env 
5. source env/bin/activate
6. pip3.6 install guppy3 cython setuptools
7. pip3.6 install --upgrade setuptools
8. pip3.6 install -r requirements.txt
________________


local_recommender
This plugin is used to generate recommended courses for a user.


# Details
This plugin does not actually do the machine learning. That is handled externally. This is because the resources required to carry out the machine learning is far more than what’s needed for the day-to-day functions of a web server.


# Configuration of dataset
* We are only using course_completion data that meet the following criteria:
    * For a user that has logged in within the last 365 days
    * The course was completed within the last 365 days
* E.g. user1 completed course1 200 days ago - This record will be put into the training dataset
* E.g. user1 completed course2 366 days ago - This record will not be put into the training dataset because the course was not completed within the last 365 days.
* E.g. user2 last logged in 366 days ago - All course completion records for this user will be excluded from the training data.
    * If user2 logs in, then the next time training data is generated then relevant records will be included.
* These lengths can be configured in the UI via this plugin's settings.
* Reasoning:
    * We needed to cut down the training dataset which is why it only uses course_completion data.


# Architecture
1.  This plugin runs a scheduled task called generate_data.
    * This generates the dataset that will be used for training.
    * The dataset is written to a file and placed in a directory. The directory is also shared by another server.
2. On an external server (VM):
    * The VM is spun up. 
    * It will read the dataset generated in 1 by looking in the shared directory.
    * Scripts will run that carries out the machine learning.
    * At time of writing, it carries out 100 epochs (iterations), and each epoch takes 16minutes.
    * You will only see output in the console at the completion of each epoch.
        * This means that it will take around 28 hours to finish running.
        * Once the training is finished it will write the top 20 recommended courses for each user into a csv file in the mounted directory.
3. This plugin runs a scheduled task called parse_data.
    * This reads the file generated in 2. from the shared directory and writes the results into a table.
4. To get the recommended courses for a user you use the recommendation_engine class which is intended to serve as the API for other plugins.


# Dataset details
* The dataset is a csv with the following columns:
    * userid, courseid, value
* The current model is JCA and this requires that the value column be a boolean.


# Adding to the dataset
* If you want to add to the dataset then you can do so by appending your values to the generated csv (named completiondata.csv)
* Your appended data must be in the same format (userid, courseid, boolean value)


# Changing the model
* If you want to change the model to something other than JCA then you can. See the documentation for the machine learning library.
________________


block_recommended_courses
This is a block developed for Unicef which displays recommended courses to a user.
* Unicef specific. It relies on a block class found in local/agora


Recommendations are generated by a separate plugin called local_recommender.
