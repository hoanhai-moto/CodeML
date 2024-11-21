from marshmallow import Schema, fields

class UploadDBInput(Schema):
    song_name = fields.Str(required=True)
    file_hash = fields.Str(required=True)