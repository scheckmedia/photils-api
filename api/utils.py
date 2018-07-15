class ApiException(Exception):
    code = 400

    def __init__(self, message, code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if code is not None:
            self.code = code
        self.payload = payload

    def __str__(self):
        return self.message

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv