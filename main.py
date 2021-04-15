import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool


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

        '''
        The JCA model appears to reference users by their position in the dataset
        rather than the value.
        E.g. position1 has userid 5546, then to get the results for userid 5546 you
        will need to use users = [1]
        '''
        users = dataset.userids.values()

        '''
        Make the predictions.

        The position of the rating corresponds to the item in the same position in dataset.itemids.

        By passing None as the second param, it gets the ratings for all courses.
        '''
        ratings = model.predict(users, None)

        # Create a csv file to write the results to. Overwrite the old one if it exists.
        f = open('dataset/ratings.csv', 'w')
        f.write('userid,courseid,rating\n')

        # Open the file in append mode so add data.
        f = open('dataset/ratings.csv', 'a')

        # Invert the key-value pair of these dicts because they are the wrong way around.
        item_ids = {value: key for key, value in dataset.itemids.items()}
        user_ids = {value: key for key, value in dataset.userids.items()}

        # Iterate through the list of users.
        for user_index in users:

            # This is the ratings for the user.
            user_ratings = ratings[user_index]

            # Create a list of tuples so we can sort by rating.
            tuple_list = []
            for i in range(len(user_ratings)):
                # This tuple is in the form (courseid, rating)
                mapped_item = (item_ids[i], user_ratings[i])
                tuple_list.append(mapped_item)

            # Sort by the user's rating in descending order.
            tuple_list.sort(key=lambda tup: tup[1], reverse=True)

            # Write the top 20 results to the file.
            for i in range(20):
                (course_id, rating) = tuple_list[i]
                f.write(f'{user_ids[user_index]},{course_id},{rating}\n')

