import subprocess
import logging
import os
import math
from datetime import timedelta, datetime


def get_output_subprocess(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    
    if exitcode != 0:
        logging.info(err)
        return []

    return out.decode('utf-8').split('\n')

def config_logging(dir):
    filename = os.path.abspath(dir) + "/default.log"
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                     datefmt='%m/%d/%Y %I:%M:%S %p')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info("Logging at: %s", filename)


def convert_to_minute(sec_list):
    return [[str(timedelta(seconds=math.floor(sec_range[0]))), str(timedelta(seconds=math.ceil(sec_range[1])))] for sec_range in sec_list]
