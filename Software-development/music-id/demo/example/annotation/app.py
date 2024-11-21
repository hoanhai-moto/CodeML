from flask import Flask, request, redirect, url_for, flash, jsonify
from preprocess_data import extract_features
import numpy as np
import pickle as p
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# cors = CORS(app)

@app.route('/', methods=['GET'])
def get_tasks():
    print('done')
    return render_


if __name__ == '__main__':
    get_tasks()
#   app.run(host = 'localhost',port=5005)