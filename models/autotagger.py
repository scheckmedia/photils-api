from nearpy import Engine
from nearpy.filters import VectorFilter
from nearpy.distances import ManhattanDistance
from nearpy.hashes import RandomBinaryProjections
import json
import numpy as np
import operator
import gzip

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
        with gzip.GzipFile('data/feature_list.json.gz') as f:
            json_bytes = f.read()
            json_str = json_bytes.decode('utf-8')
            feature_list: dict = json.loads(json_str)

        rbp = RandomBinaryProjections('rbp', 10)
        dist = ManhattanDistance()
        self.engine = Engine(self.DIMENSIONS, lshashes=[rbp], distance=dist, fetch_vector_filters=[FeatureUniqueFilter()])

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
