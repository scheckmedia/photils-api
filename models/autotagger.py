from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import operator
import json
from PIL import Image
from io import BytesIO
import base64
from api.utils import ApiException
import logging


class AutoTagger:
    DIMENSIONS = 256
    TOP_K = 30

    def __init__(self):
        self.input_shape = (256, 256)
        self.logger = logging.getLogger('photils')
        self.logger.info('load dataset and features')
        self.df = pd.read_csv('data/dataset.csv.gz')
        self.features = np.load('data/pca_features.npy')
        self.ann = NearestNeighbors(n_neighbors=self.TOP_K, algorithm='ball_tree', metric='l1', n_jobs=-1)
        self.ann.fit(self.features)

        self.valid_tags = set()
        with open('data/tags_flattened.json') as f:
            self.valid_tags.update(json.load(f))

        self.logger.info('init done')

    def init_model(self):
        import keras.backend as K
        from keras.models import load_model
        self.logger.info("load model")
        self.model = load_model('data/model.hdf5')
        self.logger.info("model warmup")
        # self.model.predict(np.zeros((1,) + self.input_shape + (3,)))  # warmup
        self.session = K.get_session()

    def get_tags(self, query: np.array):
        self.logger.info('get tags by query')
        start = datetime.now()
        items = self.ann.kneighbors(query, return_distance=False)

        tags_response = []
        for item in items:
            recommended_tags = {}
            for i in item:
                tags = []
                for tag in self.df.iloc[i]['tags'].split(' '):
                    tag = tag.replace('+', '').replace(' ', '')
                    if tag in self.valid_tags:
                        tags.append(tag)

                for tag in tags:
                    if not tag in recommended_tags:
                        recommended_tags[tag] = 0
                    recommended_tags[tag] += 1

            filtered = filter(
                lambda x: x[1] > 1, sorted(recommended_tags.items(), key=operator.itemgetter(1), reverse=True)
            )
            recommended_tags = list(map(lambda x: x[0], filtered))
            e = (datetime.now() - start).total_seconds()
            self.logger.info('found %d tags in %.2f sec' % (len(recommended_tags), e))
            tags_response += [recommended_tags]

        if items.shape[0] == 1:
            return tags_response[0]
        else:
            return tags_response

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
            prediction = self.model.predict(x).flatten()

        self.logger.info('prediction successful')

        return prediction
