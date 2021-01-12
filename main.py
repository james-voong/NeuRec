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
        userids = [
            [1,0],
            [2,1],
        ]

        itemids = [
            [18,0],
            [67,1],
        ]
        '''

        
        itemids = []
        with open("./dataset/_tmp_ml-100k/ml-100k_ratio_u0_i0.item2id") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                itemids.append(row)
        
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        #dumpme = model.evaluate()

        userids = 1
        #itemids = 18
        dumpme = model.predict(userids, itemids)
        pprint(dumpme);
