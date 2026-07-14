from http import HTTPStatus


class APIException(Exception):
    def __init__(self, status_code=HTTPStatus.INTERNAL_SERVER_ERROR, msg=None, detail=None):
        self.status_code = int(status_code)
        self.msg = msg or "Internal Server Error"
        self.detail = detail or self.msg
        super().__init__(self.detail)


class UnsupportedAudioException(APIException):
    def __init__(self, detail=None):
        super().__init__(
            status_code=HTTPStatus.BAD_REQUEST, msg="Unsupported audio", detail=detail
        )


class TranscriptionException(APIException):
    def __init__(self, detail=None):
        super().__init__(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            msg="Transcription failed",
            detail=detail,
        )
