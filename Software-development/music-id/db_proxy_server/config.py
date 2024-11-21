import os, sys

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(basedir)

import music_id.configs as configs

config = configs.init()

database_conf = config.get('database')
