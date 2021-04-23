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
