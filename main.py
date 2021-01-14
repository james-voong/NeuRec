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
            print("if")
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            print("elif")
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            print("else")
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)
        '''
        # userids is an array of userid numbers that we want to predict.
        userids = [
            1,
            2,
            3,
        ]

        # still trying to understand what these are supposed to be.
        itemids = [
            [18,0],
            [67,1],
        ]
        '''

        '''
        itemids = []
        with open("./dataset/_tmp_ml-100k/ml-100k_ratio_u0_i0.item2id") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                itemids.append(row)
        '''

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()

        # Userids of the user you want to predict.
        users = [244]

        # Make the predictions.
        results = model.predict(users);

        for key, value in enumerate(results[0]):
            print(f"Key is {key} and the value is {value}")

        # We need to get the index for each user to look them up
        for user_index, user in enumerate(users):

            # This is the user data for each user.
            user_data = results[user_index]

            # And we need the item index.
            # Starts from 1 because numpy says so.
            for item_index, value in enumerate(user_data, start=1):

                #  Using the indexes to work things out, we can list all the scored for each user.
                print(f"User {user} has score {value} for item {dataset.itemids[item_index]}")
