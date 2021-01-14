import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool

from pprint import pprint
import csv


np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
if __name__ == "__main__":
    conf = Configurator("NeuRec.properties", default_section="hyperparameters")
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    # num_thread = int(conf["rec.number.thread"])

    # if Tool.get_available_gpus(gpu_id):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        dumpme = model.evaluate()
        pprint(dumpme);
'''
'''
conf = Configurator("NeuRec.properties", default_section="hyperparameters")
gpu_id = str(conf["gpu_id"])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

recommender = conf["recommender"]
my_module = importlib.import_module("model.general_recommender." + recommender)

dataset = Dataset(conf)
config = tf.ConfigProto()

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]

userids = [
    "1, 0",
    "2, 1",
]

with tf.Session(config=config) as sess:
    MyClass = getattr(my_module, recommender)
    model = MyClass(sess, dataset, conf)
    sess.run(tf.global_variables_initializer())
    #model.build_graph()
    model.train_model()

    model.predict(userids)
    pprint(model)
'''
'''
userids = []
with open("./dataset/_tmp_ml-100k/ml-100k_ratio_u0_i0.user2id") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        userids.append(row)

pprint(userids)
'''

myarr = [
    'string1',
    'string2',
    'string3',
    'string4',
]

for count, value in enumerate(myarr):
    print(count, value)

for i in range(0, len(myarr)):
    print(i, myarr[i])

# Make the predictions
results = model.predict([201])

# We need to get the index for each user to look them up
for user_index, user in enumerate(users):

    # This is the user data for each user.
    user_data = results[user_index]

    # And we need the item index.
    # Unsure if this is right.
    # Starts from 1 because numpy
    for item_index, value in enumerate(user_data, start=1):

        # Using the indexes to work things out, we can list all the scored for each user.
        print(f"User {user} has score {value} for item {dataset.itemids[item_index]}")
