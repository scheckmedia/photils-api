from nearpy import Engine
from nearpy.filters import VectorFilter, NearestFilter
from nearpy.distances import ManhattanDistance
from nearpy.hashes import RandomBinaryProjections
import numpy as np
import pandas as pd
import operator
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
from api.utils import ApiException
import logging


class FeatureUniqueFilter(VectorFilter):
    def __init__(self):
        pass

    def filter_vectors(self, input_list):
        """
        Returns subset of specified input list.
        """
        unique_dict = {}
        for v in input_list:
            unique_dict[v[1]['id']] = v
        return list(unique_dict.values())


class AutoTagger:
    DIMENSIONS = 256
    NUM_HASH_TABLES = 16
    HASH_LENGTH = 8
    TOP_K = 10

    def __init__(self):
        self.logger = logging.getLogger('photils')
        self.logger.info('load dataset and features')
        df = pd.read_csv('data/dataset.csv.gz')
        features = np.load('data/pca_features.npy')

        valid_tags = set()
        with open('data/tags_flattened.json') as f:
            valid_tags.update(json.load(f))

        self.logger.info('initialize LSH engine')

        rbps = []
        for i in range(0, self.NUM_HASH_TABLES + 1):  # number of hash tables
            rbps += [RandomBinaryProjections('rbp_%d' % i, self.HASH_LENGTH)]

        dist = ManhattanDistance()
        nearest = [NearestFilter(self.TOP_K)]
        fetch = [FeatureUniqueFilter()]
        self.engine = Engine(self.DIMENSIONS, lshashes=rbps,
                             distance=dist, vector_filters=nearest,  fetch_vector_filters=fetch)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            feature = np.array(features[idx], dtype=np.float32)
            tags = []
            for tag in row['tags'].split(' '):
                tag = tag.replace('+', '').replace(' ', '')
                if tag in valid_tags:
                    tags.append(tag)

            self.engine.store_vector(feature, {'tags': tags, 'id': row['id']})

        self.logger.info('LSH engine initialization successful')

    def init_model(self):
        import keras.backend as K
        from keras.models import load_model
        self.logger.info("load model")
        self.input_shape = (256, 256)
        self.model = load_model('data/model.hdf5')
        self.logger.info("model warmup")
        # self.model.predict(np.zeros((1,) + self.input_shape + (3,)))  # warmup
        self.session = K.get_session()

    def get_tags(self, query: np.array):
        self.logger.info('get tags by query')
        recommended_tags: dict = {}
        for feature in self.engine.neighbours(query):
            for tag in feature[1]['tags']:
                recommended_tags.setdefault(tag, 0)
                recommended_tags[tag] += 1

        # filtered = filter(lambda x: x[1] > 1,
        #                   sorted(recommended_tags.items(), key=operator.itemgetter(1), reverse=True))
        filtered = sorted(recommended_tags.items(), key=operator.itemgetter(1), reverse=True)
        recommended_tags = list(
            map(lambda x: x[0], filtered)
        )

        self.logger.info('found %d items' % len(recommended_tags))

        return recommended_tags

    def get_feature(self, base64img):
        self.logger.info('get feature from image')
        try:
            img = Image.open(BytesIO(base64.b64decode(base64img)))
            img = img.resize(self.input_shape, Image.BICUBIC).convert('RGB')
        except Exception:
            self.logger.error('image loading fail')
            raise ApiException("invalid base64 image", 400)

        from keras.applications.resnet50 import preprocess_input
        from keras.preprocessing import image

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        self.logger.info('run prediction')
        with self.session.as_default():
            x = preprocess_input(x)
            prediction = self.model.predict(x).reshape((2048,))

        self.logger.info('prediction successful')

        return prediction
