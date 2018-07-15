from flask import Blueprint, request, jsonify
from .utils import ApiException
from models.autotagger import AutoTagger

import numpy as np

api = Blueprint('auto_tagger_api', 'auto_tagger_api')
tagger = AutoTagger()


@api.route('/tags/', methods=['POST'])
def get_tags_by_feature():
    data = request.get_json()
    if 'feature' not in data:
        raise ApiException("invalid feature parameter", 400)

    if len(data['feature']) != tagger.DIMENSIONS:
        raise ApiException("invalid dimension of feature vector", 400)

    query = np.array(data['feature'])
    recommended_tags = tagger.get_tags(query)

    return jsonify({'tags': recommended_tags, 'success': True})

