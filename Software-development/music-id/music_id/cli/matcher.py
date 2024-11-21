### Example:
#
# For youtube link:
#       python music_id/cli/matcher.py -c 240 -y 'https://www.youtube.com/watch?v=tKYeJfboirdfaQ'
#
# For a audio file:
#       python music_id/cli/matcher.py -c 240 -f '/path/to/audio/file.mp3'
#
# ================================

import os
import sys
import click
import subprocess
import datetime
import time
import glob
import logging
import warnings
import json
import psutil
import signal
from timeloop import Timeloop
import traceback
import uuid


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from music_id import MusicID, __clear_cache
from music_id.tools.util import config_logging

tl = Timeloop()
process_measure = []
sys_info_path = ''

EXTENSION = 'wav'


###


@tl.job(interval=datetime.timedelta(seconds=2))
def measure_system_parameters():
    global process_measure
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0]
        cpu = process.cpu_percent(interval=2)
        process_measure.append('%d, %0.4f, %0.4f' % (2 * (len(process_measure) + 1), cpu, mem))
    except:
        print(traceback.format_exc())

def log_sys_info():
    global sys_info_path, process_measure
    with open(sys_info_path, 'w') as outfile:
        outfile.write('Time step (s), CPU (%), RAM (bytes)\n')
        outfile.write("\n".join(process_measure))
    del process_measure[:]


def run(ylink=None, 
        input_file=None,
        outdir="/tmp/music-id-" + uuid.uuid4().__str__(), 
        chunk=600, 
        delete_after=0, 
        library='audfprint'):
    '''
    ylink: Youtube URL
    '''
    print("Output at: %s" % outdir)

    try:
        stime = time.time()
        now_str = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
        outdir = os.path.join(os.path.abspath(outdir), now_str)
        result_dir = os.path.join(outdir, "results")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(result_dir)

        config_logging(outdir)

        if input_file is not None:
            audio_path = os.path.abspath(input_file)
            # audio_paths = [input_file] 

        elif ylink is not None:
            # Download all youtube link with audio format is mp3
            logging.info("Downloading...")
            dtime = time.time()
            dl_command = ["youtube-dl", "-x",
                        "-o", outdir + "/%(title)s.%(ext)s",
                        "--audio-format", EXTENSION,
                        ylink]
            subprocess.call(dl_command)
            logging.info("Download completed in %d (s)", time.time() - dtime)

            __clear_cache()
            audio_paths = glob.glob(outdir + ("/*." + EXTENSION))
            if len(audio_paths) > 0:
                audio_path = audio_paths[0]
            else:
                raise Exception("Audio file not found!")

        # Detect on all audio
        # core_library = (CoreLibEnum.AUDIFPRINT if library == 'audfprint' else CoreLibEnum.DEJAVU)
        music_id = MusicID()
        global sys_info_path

        # Processing Audio File
        logging.info('Processing for %s', audio_path)

        song_name = os.path.splitext(os.path.basename(audio_path))[0]
        sys_info_path = os.path.join(result_dir, song_name + '.sys_info.csv')
        out_path = os.path.join(result_dir, song_name + ".annotation.json")

        # Detect music by fingerprint
        annotations = music_id.detect(audio_path, chunk=chunk)

        result = {
            'total_time': time.time() - stime,
            'config': music_id.config.get('audfprint', {}),
            'result': annotations
        }

        # Dump annotations to file
        with open(out_path, 'w') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
            
        logging.info("The result stored at %s", out_path)

        # Log system information
        log_sys_info()

        # Stop time loop
        # tl.stop()
        logging.info("All processes completed in %d(s)", time.time() - stime)
    except:
        logging.info(traceback.format_exc())
    finally:
        if delete_after == 1:
            try:
                for file in audio_paths:
                    os.remove(file)
            except: 
                pass
        logging.info('The process ended.')
        
    return result


def handle_process_terminate(signalNumber, frame):
    logging.info('The process was terminated by signal number %d', signalNumber)
    log_sys_info()


@click.command()
@click.option("-y", "--ylink", default=None, help='A youtube link')
@click.option("-f", "--input_file", default=None, help='A input file path')
@click.option("-o", "--outdir", default="/tmp/music-id-" + uuid.uuid4().__str__(), help="Output directory.")
@click.option("-c", "--chunk", default=600, help="split large audio into multi chunk by second number.")
@click.option("-d", "--delete_after", default=0, help="Delete input after the process complete.")
@click.option('-lib', '--library', default='audfprint', help='What the library core used for building a database. "audfprint" or "dejavu".')
def main(ylink, input_file, outdir, chunk, delete_after, library):
    run(ylink, input_file, outdir, chunk, delete_after, library)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # tl.start()
    signal.signal(signal.SIGHUP, handle_process_terminate)
    signal.signal(signal.SIGINT, handle_process_terminate)
    signal.signal(signal.SIGQUIT, handle_process_terminate)
    signal.signal(signal.SIGILL, handle_process_terminate)
    signal.signal(signal.SIGTRAP, handle_process_terminate)
    signal.signal(signal.SIGABRT, handle_process_terminate)
    signal.signal(signal.SIGBUS, handle_process_terminate)
    signal.signal(signal.SIGFPE, handle_process_terminate)
    #signal.signal(signal.SIGKILL, handle_process_terminate)
    signal.signal(signal.SIGUSR1, handle_process_terminate)
    signal.signal(signal.SIGSEGV, handle_process_terminate)
    signal.signal(signal.SIGUSR2, handle_process_terminate)
    signal.signal(signal.SIGPIPE, handle_process_terminate)
    signal.signal(signal.SIGALRM, handle_process_terminate)
    signal.signal(signal.SIGTERM, handle_process_terminate)
    main()
