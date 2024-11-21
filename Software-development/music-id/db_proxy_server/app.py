import os, sys

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(basedir + '/../')

from gevent import monkey
monkey.patch_all()

from flask import request, Flask, jsonify, abort
from werkzeug.exceptions import BadRequest
from marshmallow import Schema, fields
from flask_cors import CORS
from db_proxy_server.db_handle import *
from gevent.pywsgi import WSGIServer
from schemas.upload_db_input import UploadDBInput
import time
import click
import traceback
import json
import logging
# from celery import Celery
import pickle
import zlib

log_file = basedir + '/logs/default.log'
logging.basicConfig(filename=log_file,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


from music_id import configs
from music_id.core.dejavu import Dejavu

config = configs.init()

app = Flask(__name__)
CORS(app)

# app.config.update(
#     CELERY_BROKER_URL='redis://localhost:6379',
#     CELERY_RESULT_BACKEND='redis://localhost:6379'
# )


upload_db_input = UploadDBInput()


# def make_celery(app):
#     celery = Celery(
#         app.import_name,
#         backend=app.config['CELERY_RESULT_BACKEND'],
#         broker=app.config['CELERY_BROKER_URL']
#     )
#     celery.conf.update(app.config)

#     class ContextTask(celery.Task):
#         def __call__(self, *args, **kwargs):
#             with app.app_context():
#                 return self.run(*args, **kwargs)

#     celery.Task = ContextTask
#     return celery
###

# celery = make_celery(app)
global dejavu
dejavu = None

# @celery.task()
def insert_to_db(song_name, file_hash, hashes):
    global dejavu
    dejavu.insert_fingerprint_db(song_name, file_hash, hashes)

@app.route('/api/db_proxy/test', methods=['GET'])
def test():
    return 'Successes!'

@app.route('/api/db_proxy/upload_db', methods=['POST'])
def upload_db():
    if request.headers.get('Content-Type') != "application/octet-stream":
        raise BadRequest('Header must be the application/octet-stream')

    json_data = pickle.loads(zlib.decompress(request.input_stream.read()))

    errors = upload_db_input.validate(json_data)
    if errors:
        raise BadRequest(errors)

    global dejavu
    if dejavu is None:
        db_info = json_data.get('db_info')
        conf = {
            "database_type": json_data.get('db_type'),
            "database": {
                "host": config.get('database').get('host'),
                "user": db_info.get('user'),
                "password": db_info.get('password'),
                "database": db_info.get('database'),
            }
        }
        dejavu = Dejavu(conf)

    song_name = json_data.get('song_name')
    logging.info('Request song: %s', song_name)

    file_hash = json_data.get('file_hash')

    hashes = json_data.get('hashes')

    try:
        s_time = time.time()
        insert_to_db(song_name, file_hash, hashes)
        # result = insert_to_db.delay(song_name, file_hash, hashes)
        # result.wait()
    except:
        logging.info('Error: %s', traceback.format_exc())
        return {
            'Status': 'An unexpected error has occured.'
        }, 200
    else:
        insert_time = time.time() - s_time
        logging.info('''Inserted successfully:
                song_name: %s,
                time: %s(s)
        ''', song_name, insert_time)
        return {
            "Song_name": song_name,
            "Insert_db_time": insert_time,
            "Time_unit": "second",
            "Status": "Successed."
        }, 200


@click.command()
@click.option('--port', default=5000, help='Port number')
def main(port):
    print('Running at port %d' % port)
    # app.run(port=port)
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()


if __name__ == "__main__":
    main()


###
