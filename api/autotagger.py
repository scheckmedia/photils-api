from flask import Blueprint, request, jsonify, current_app
from .utils import ApiException
import base64
import numpy as np

api = Blueprint('auto_tagger_api', 'auto_tagger_api')


@api.route('/tags', methods=['POST'])
def get_tags_by_feature():
    tagger = current_app.tagger
    data = request.get_json()
    key = None
    allowed_keys = ['feature', 'features', 'image']

    for k in data.keys():
        if k in allowed_keys:
            key = k
            break

    if key is None:
        raise ApiException("invalid feature parameter", 400)

    if key == 'feature':
        if not isinstance(data[key], list):
            feature = np.frombuffer(base64.decodebytes(str.encode(data[key])), dtype=np.float32)
        else:
            feature = np.array(data[key])

        if feature.shape[-1] != tagger.DIMENSIONS:
            raise ApiException("invalid dimension of feature vector", 400)

        query = [np.array(feature)]
    elif key == 'features':
        if not len(data[key]):
            raise ApiException("empty request", 400)

        features = []
        for feature in data[key]:
            if not isinstance(feature, list):
                feature = np.frombuffer(base64.decodebytes(str.encode(feature)), dtype=np.float32)
            else:
                feature = np.array(feature)

            if feature.shape[-1] != tagger.DIMENSIONS:
                raise ApiException("invalid dimension of feature vector", 400)

            features += [feature]

        query = np.array(features)
    else:
        query = [tagger.get_feature(data['image'])]

    recommended_tags = tagger.get_tags(query)


    return jsonify({'tags': recommended_tags, 'success': True})


