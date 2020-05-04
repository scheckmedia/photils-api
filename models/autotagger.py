import tensorflow as tf
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import operator
import json
from PIL import Image
from io import BytesIO
import base64
from api.utils import ApiException
import logging


class AutoTagger:
    DIMENSIONS = 128
    TOP_K = 30

    def __init__(self):
        self.input_shape = (224, 224)
        self.logger = logging.getLogger('photils')
        self.logger.info('load dataset and features')
        self.features = np.load('data/features.npy')
        self.tagids = np.load('data/tagids.npy', allow_pickle=True)
        self.model = tf.lite.Interpreter('data/model.tflite')
        self.model.allocate_tensors()

        self.logger.info('fit knn')
        self.ann = NearestNeighbors(
            n_neighbors=self.TOP_K, algorithm='ball_tree', metric='l2', n_jobs=-1)
        self.ann.fit(self.features)

        with open('data/tags_flattened.json') as f:
            self.valid_tags = json.load(f)

        self.logger.info('init done')

    def get_tags(self, query: np.array, k: int = None):
        self.logger.info('get tags by query')
        start = datetime.now()
        kwargs = {'return_distance': False}

        if k is not None and k > 0:
            kwargs['n_neighbors'] = k

        query_indices = self.ann.kneighbors(query, **kwargs)

        tags_response = []
        for indices in query_indices:
            tagid_indices = self.tagids[indices]
            recommended_tags = {}
            for tag_index in tagid_indices:
                for tagid in tag_index:
                    tag = self.valid_tags[tagid]
                    recommended_tags[tag] = recommended_tags.get(tag, 0) + 1

            filtered = sorted(recommended_tags.items(),
                              key=operator.itemgetter(1), reverse=True)

            recommended_tags = list(map(lambda x: x[0], filtered))
            e = (datetime.now() - start).total_seconds()
            self.logger.info('found %d tags in %.2f sec' %
                             (len(recommended_tags), e))
            tags_response += [recommended_tags]

        if query_indices.shape[0] == 1:
            return tags_response[0]
        else:
            return tags_response

    def get_feature(self, base64img):
        self.logger.info('get feature from image')
        try:
            img = self.process(base64img)
        except Exception as ex:
            self.logger.error('image loading fail\n {}'.format(ex))
            raise ApiException("invalid base64 image", 400)

        self.logger.info('run prediction')
        prediction = self.predict(img)
        prediction = prediction / \
            np.linalg.norm(prediction, 2, axis=1, keepdims=True)
        prediction = np.squeeze(prediction)
        self.logger.info('prediction successful')

        return prediction

    def process(self, base64img):
        img = Image.open(BytesIO(base64.b64decode(base64img)))
        img = np.array(img.convert('RGB'))
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, self.input_shape)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image):
        input_index = self.model.get_input_details()[0]["index"]
        output_index = self.model.get_output_details()[0]["index"]
        self.model.set_tensor(input_index, image)
        self.model.invoke()
        return self.model.get_tensor(output_index)
