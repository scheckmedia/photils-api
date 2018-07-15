from flask import Flask, jsonify
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException

from api.autotagger import api as auto_tagger_api
from api.utils import ApiException

def run_server(name):
    def make_json_error(ex):
        print(ex)
        response = jsonify(message=str(ex), success=False)
        response.status_code = (ex.code
                                if isinstance(ex, HTTPException) or isinstance(ex, ApiException)
                                else 500)
        return response

    app = Flask(name)

    for code in default_exceptions.keys():
        app.errorhandler(code)(make_json_error)

    app.register_blueprint(auto_tagger_api)
    app.run(host='0.0.0.0', port=11000, threaded=True, debug=False)
