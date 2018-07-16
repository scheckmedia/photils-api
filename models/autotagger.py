from nearpy import Engine
from nearpy.filters import VectorFilter, NearestFilter
from nearpy.distances import ManhattanDistance
from nearpy.hashes import RandomBinaryProjections
import keras
import keras.backend as K
from keras_applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import json
import numpy as np
import operator
import gzip
from PIL import Image
from io import BytesIO
import base64
from api.utils import ApiException

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
    DIMENSIONS = 64

    def __init__(self):
        self.input_shape = (256, 256)
        self.model = ResNet50(weights='imagenet', pooling='avg', include_top=False, input_shape=self.input_shape + (3,))
        self.model.predict(np.zeros((1,) + self.input_shape + (3,)))  # warmup
        self.session = K.get_session()
        self.graph = K.tf.get_default_graph()
        self.graph.finalize()

        with open('data/pca_components.json', 'r') as f:
            self.pca_componentes = np.array(json.load(f))

        with gzip.GzipFile('data/feature_list.json.gz') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            feature_list: dict = json.loads(json_str)

        rbp = RandomBinaryProjections('rbp', 8, rand_seed=42)
        dist = ManhattanDistance()
        nearest = [NearestFilter(10)]
        fetch = [FeatureUniqueFilter()]
        self.engine = Engine(self.DIMENSIONS, lshashes=[rbp],
                             distance=dist, vector_filters=nearest,  fetch_vector_filters=fetch)

        for photo_id, item in feature_list.items():
            feature = np.array(item['feature'])
            tags = item['meta']['tags']
            self.engine.store_vector(feature, {'tags': tags, 'id': photo_id})

    def get_tags(self, query: np.array):
        recommended_tags: dict = {}
        for feature in self.engine.neighbours(query):
            for tag in feature[1]['tags']:
                recommended_tags.setdefault(tag, 0)
                recommended_tags[tag] += 1

        recommended_tags = list(
            map(lambda x: x[0], sorted(recommended_tags.items(), key=operator.itemgetter(1), reverse=True))
        )

        return recommended_tags

    def get_feature(self, base64img):
        try:
            img = Image.open(BytesIO(base64.b64decode(base64img))).resize(self.input_shape).convert('RGB')
        except Exception:
            raise ApiException("invalid base64 image", 400)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        with self.session.as_default():
            with self.graph.as_default():
                x = preprocess_input(x)
                prediction = self.model.predict(x).reshape((2048,)).astype(np.float16)

        return prediction.dot(self.pca_componentes)
