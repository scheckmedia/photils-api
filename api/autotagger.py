from flask import Blueprint, request, jsonify, current_app
from .utils import ApiException

import numpy as np

api = Blueprint('auto_tagger_api', 'auto_tagger_api')


@api.route('/tags', methods=['POST'])
def get_tags_by_feature():
    tagger = current_app.tagger
    data = request.get_json()
    if 'feature' not in data and 'image' not in data:
        raise ApiException("invalid feature parameter", 400)

    if 'feature' in data:
        if len(data['feature']) != tagger.DIMENSIONS:
            raise ApiException("invalid dimension of feature vector", 400)

        query = np.array(data['feature'])
    else:
        query = tagger.get_feature(data['image'])

    recommended_tags = tagger.get_tags(query)


    return jsonify({'tags': recommended_tags, 'success': True})


