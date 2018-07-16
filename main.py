from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from log import setup_custom_logger
from api.utils import ApiException


def make_json_error(ex):
    response = jsonify(message=str(ex), success=False)
    response.status_code = (ex.code
                            if isinstance(ex, HTTPException) or isinstance(ex, ApiException)
                            else 500)
    return response

logger = setup_custom_logger('photils')
logger.info("initialize flask")

app = Flask(__name__)
CORS(app)




for code in default_exceptions.keys():
    app.errorhandler(code)(make_json_error)


from api.autotagger import api as auto_tagger_api
app.register_blueprint(auto_tagger_api)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11000, threaded=True, debug=False)
