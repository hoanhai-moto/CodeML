
import os, sys

cli_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cli_dir + "/../../")
sys.path.append(cli_dir + "/../core")

import warnings
import json
import os
warnings.filterwarnings("ignore")
import timeit
from music_id.core.dejavu import Dejavu
from music_id.core.dejavu.logic.recognizer.file_recognizer import FileRecognizer
import argparse
import logging
import os

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True,
   help="Input folder mp3")
args = vars(ap.parse_args())

log_dir = cli_dir + '/../logs'
logging.basicConfig(filename=log_dir + '/default_%d.log' % (len(os.listdir(log_dir))), 
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

# load config from a JSON file (or anything outputting a python dictionary)
with open(cli_dir + "/../configs/config_default.json") as f:
    config = json.load(f)

start = timeit.default_timer()

if __name__ == '__main__':
  data_dir = os.getcwd()
  data_dir = os.path.join(data_dir,"")
  my_path = os.path.abspath(os.path.dirname(data_dir))
  # create a Dejavu instance
  djv = Dejavu(config)
  # Fingerprint all the mp3's in the directory we give it
  djv.fingerprint_directory(os.path.join(my_path,args['input']), [".mp3", ".wav"])

stop = timeit.default_timer()

msg = 'Time total import: ' + str(stop - start)
print(msg)  
logging.info(msg)
